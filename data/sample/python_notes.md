# Python Programming Notes

## Why Python for AI/ML?
Python dominates AI and data science because of its readable syntax, massive ecosystem (NumPy, Pandas, PyTorch, TensorFlow), and fast iteration cycle. You write less code to do more.

## Core Data Types

### Lists
Ordered, mutable sequences. Use for collections that change.
```python
items = [1, 2, 3]
items.append(4)       # [1, 2, 3, 4]
items[0]              # 1  (zero-indexed)
items[-1]             # 4  (last element)
items[1:3]            # [2, 3]  (slicing)
```

### Dictionaries
Key-value pairs. Use for structured data and fast lookups.
```python
person = {"name": "Alice", "age": 30}
person["name"]        # "Alice"
person.get("email", "N/A")  # "N/A"  (safe lookup with default)
```

### Tuples
Ordered, immutable. Use for fixed collections (coordinates, RGB values).
```python
point = (10, 20)
x, y = point          # unpacking
```

### Sets
Unordered, unique elements. Use for deduplication and membership tests.
```python
tags = {"python", "ml", "python"}   # {"python", "ml"}
"ml" in tags          # True  (O(1) lookup)
```

## Functions

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

greet("Alice")              # "Hello, Alice!"
greet("Bob", "Hi")          # "Hi, Bob!"
```

### Lambda Functions
Anonymous one-liner functions.
```python
square = lambda x: x ** 2
square(5)   # 25

# Common use: sorting
names = ["Charlie", "Alice", "Bob"]
names.sort(key=lambda n: len(n))  # sort by length
```

### *args and **kwargs
```python
def log(*args, **kwargs):
    print(args)    # tuple of positional args
    print(kwargs)  # dict of keyword args

log(1, 2, 3, level="info", tag="ml")
```

## List Comprehensions
Concise way to build lists. Faster than for loops.
```python
squares = [x**2 for x in range(10)]
evens   = [x for x in range(20) if x % 2 == 0]
matrix  = [[i*j for j in range(3)] for i in range(3)]
```

## Classes and OOP

```python
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species

    def speak(self):
        return f"{self.name} makes a sound"

    def __repr__(self):
        return f"Animal({self.name}, {self.species})"

class Dog(Animal):
    def speak(self):           # override parent method
        return f"{self.name} barks!"

dog = Dog("Rex", "Canis lupus")
dog.speak()   # "Rex barks!"
```

## Error Handling

```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
except (TypeError, ValueError) as e:
    print(f"Bad input: {e}")
finally:
    print("Always runs")   # cleanup code
```

## File I/O

```python
# Write
with open("notes.txt", "w") as f:
    f.write("Hello, world!")

# Read
with open("notes.txt", "r") as f:
    content = f.read()

# Read line by line (memory efficient for large files)
with open("big_file.txt") as f:
    for line in f:
        process(line.strip())
```

## Python for Data Science

### NumPy
Fast array operations. Foundation of all ML libraries.
```python
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a + b          # [5, 7, 9]  (element-wise)
a.dot(b)       # 32  (dot product)
a.reshape(3,1) # column vector
np.zeros((3,3))
np.random.randn(100, 10)  # random matrix
```

### Pandas
Data manipulation and analysis.
```python
import pandas as pd
df = pd.read_csv("data.csv")
df.head()                    # first 5 rows
df["column"].mean()          # column statistics
df[df["age"] > 30]           # filter rows
df.groupby("city").mean()    # aggregate
df.dropna()                  # remove missing values
df.fillna(0)                 # fill missing values
```

## Virtual Environments
Always use a virtual environment to isolate project dependencies.

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip freeze > requirements.txt
deactivate
```

## Common Gotchas

1. **Mutable default arguments**
```python
# WRONG — list is shared across all calls
def add(item, lst=[]):
    lst.append(item)
    return lst

# RIGHT
def add(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst
```

2. **Integer division**
```python
7 / 2    # 3.5  (float division)
7 // 2   # 3    (integer division)
```

3. **Shallow vs deep copy**
```python
import copy
a = [[1, 2], [3, 4]]
b = a.copy()        # shallow — inner lists still shared
c = copy.deepcopy(a)  # deep — fully independent copy
```

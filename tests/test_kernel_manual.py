from orchestrator.kernel_session import KernelSession

kernel = KernelSession()

try:
    print("=" * 50)
    print("Test 1: Simple Expression")
    result = kernel.execute("2 + 2")
    print(result)

    print("\n" + "=" * 50)
    print("Test 2: print()")
    result = kernel.execute("""
print("Hello from kernel!")
""")
    print(result)

    print("\n" + "=" * 50)
    print("Test 3: Variable Persistence")
    kernel.execute("x = 100")
    result = kernel.execute("x")
    print(result)

    print("\n" + "=" * 50)
    print("Test 4: Pandas")
    result = kernel.execute("""
import pandas as pd

df = pd.DataFrame({
    "Name": ["Alice", "Bob"],
    "Age": [25, 30]
})

df
""")
    print(result)

    print("\n" + "=" * 50)
    print("Test 5: Error Handling")
    result = kernel.execute("1/0")
    print(result)

    print("\n" + "=" * 50)
    print("Test 6: Matplotlib")
    result = kernel.execute("""
import matplotlib.pyplot as plt

plt.plot([1,2,3], [4,5,6])
plt.title("Demo Plot")
plt.show()
""")

    print("Number of images:", len(result["images"]))
    print("Stdout:", result["stdout"])
    print("Error:", result["error"])

finally:
    kernel.shutdown()
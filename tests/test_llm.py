from orchestrator.kernel_session import KernelSession
from orchestrator.llm import LLM

llm = LLM()
kernel = KernelSession()

try:
    task = "Create a pandas DataFrame with names and ages and print it."

    # Step 1: Generate Python code
    code = llm.generate_code(task)

    print(repr(code))
    print("\n\n\n")
    
    
    print("=" * 60)
    print("Generated Code:")
    print("=" * 60)
    print(code)

    # Step 2: Execute generated code
    result = kernel.execute(code)
    
    # print(result)

    print("\n" + "=" * 60)
    print("Execution Result:")
    print("=" * 60)

    print("Stdout:")
    print(result["stdout"])

    print("\nResult:")
    print(result["result"])

    print("\nError:")
    print(result["error"])

    print("\nImages:")
    print(len(result["images"]))

finally:
    kernel.shutdown()
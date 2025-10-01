Task background:
Read the megafuse paper draft PDF word by word. I'm trying to implement an extensible FusionChain class in numpy to demonstrate the idea with fusion axis blocking. The solution should be in four files:
- fusion_chain.py contains the main fusion chain orchestrator
- operators.py contains the operator definitions. Use rmsnorm+matmul fusion as an example. Design appropriate operator dataclasses.
- tensors.py contains numpy ndarray wrapper class with helper functions.
- fusion_typing.py contains the data type definitions in a centralized location.
- test_fusion.py contains an example test.
In the end, I want to be able to easily define fx, a series of gb and hb functions, and FusionChain fuses those operators together according to the fused X+accumulation algorithm for me.

To do:
1. The rmsnorm+matmul test example should compute rmsnorm(V1)@V2. So the output for example, looks like:
  V1: (512, 1024)
  V2: (1024, 128)
  out = (512, 128)
Use this as debugging example.

2. The current FusionChain implementation only supports two operators.
I intended this to be more general purpose and would easily allow users to extend to multiple operators.

Coding style requirements:
1. Clearly add type hints.
2. Write concise and clear function docstring.
3. Do not leave too much inline comments in the codes.
4. Keep the codes concise, clear and readable.
5. Remove stale and unused codes where appropriate.

Development tips:
1. When testing, use ~/venvs/autotune Python venv.
2. Do not create extra files or readmes.
3. Refer to main.pdf when you have doubts about the algorithm design.
4. Think hard and reason in details for the tasks.
5. Propose changes first, analyze and iterate on your changes design, before implementing, testing and debugging.
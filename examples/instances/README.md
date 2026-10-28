# Bundled example instances

## Synthetic scheduling tutorial

`scheduling.json` is a four-job synthetic input authored for the
`examples.instance_pipeline` tutorial. It supplies a name, positive integer
processing times and nonnegative integer due dates. It is not imported from C++
and is not a competitive-performance benchmark. See the file-backed adapter guide
in `docs/problem-adapters.md` in the repository.

## Knapsack reference instances

`kp10.txt` and `kp50.txt` contain the reference numeric data from
[RKO-solver/RKO_Cpp_v2.0](https://github.com/RKO-solver/RKO_Cpp_v2.0/tree/126e35873e50dc308795b58b8c75d37f8de4c857/Instances/KP),
revision `126e35873e50dc308795b58b8c75d37f8de4c857`.
Trailing whitespace is removed and line endings are normalized; numeric data
is unchanged. The upstream MIT license is preserved in `LICENSE` in this directory.

Format: item count and capacity, followed by one profit/weight pair per item.
Independent exact 0/1 Knapsack dynamic programming gives optimum profits of
**31** and **149**, respectively. These small instances check correctness and
example usability; they are not large-scale performance benchmarks.

SHA-256:

- `kp10.txt`: `837b348da1dda53d2635d330e6c0453e7a067a2629632e4cb0e36904e18d0938`
- `kp50.txt`: `ed7240148053a3676f04bba92294357dd5bc4a53b4d585ebefc724171070c277`

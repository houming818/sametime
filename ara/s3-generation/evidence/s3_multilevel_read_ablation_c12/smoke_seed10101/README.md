# C12 Matched-Stream Smoke

Taskd jobs `178`, `179`, and `180` ran the `c10`, `read`, and `read_up`
arms on `io`. Each arm used 512 WMT training rows, 32 validation rows, 32
test rows, 20 optimizer steps and batch size 8.

The experimental identity gate passed:

```text
stream SHA-256 (all arms)
38a8fd587cd381877f22cf95ce5df69fff6e268603b95059f1e31907fdd1c552

READ initialization SHA-256 (read and read_up)
c94946a5a005345eef95b2520419a35e71fb30b630d4bc35717e09d6a80deadc
```

Train, validation and test row hashes also match exactly across all arms.

After only 20 steps, ablating coarse READ depths improves rather than harms
NLL. This reproduces the early harmful phase seen in the C11 smoke. The smoke
therefore supports the experiment wiring and falsification sensitivity, but it
does not support the mechanism claim. Formal training must determine whether
the ablation signs reverse under a matched long run.

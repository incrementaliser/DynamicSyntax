# Quick list of things to remember if you are extending the induction code

Don't forget to check these:

- The two `map`s in `tree`
  - These have to [somewhat?] match the conditions at the beginning of `getAbstractions()` in `TTRRecordType`.
- The `subj/obj/ind_obj` restrictions in lattice code.
- `getAbstractions()` x2 (one in `TTRFormula` (tree) and one in `TTRRecordType` (rt)), and the conditions at their beginning.
  - AA has added getMaximalFilteredAbstractions fyi.
- The `computational-actions.txt` file used for induction (latest available in `resource/2025-babyds-seeded-induction`)
  - Don't confuse this with the ones used for parsing (I learned this the hard way...)
- Contacting AA/AE by email!

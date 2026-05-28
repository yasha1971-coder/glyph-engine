# SA32 BOUNDARY AUDIT V1

Status:
Boundary audit before 4GB latency scaling.

Reason:
4GB corpus scaling must not be measured until suffix-array integer boundaries are understood.

## Key Finding

The current canonical build_sa path is int32-based, not safe for 4GB.

Evidence:

src/build_sa.cpp:
- checks corpus size against std::numeric_limits<int32_t>::max()
- casts corpus size to int32_t
- stores SA in std::vector<int32_t>
- calls libsais int32 API
- writes SA values as uint32_t

Practical limit:

int32 max:
2,147,483,647

Current successful 2GB corpus:
2,000,000,000 bytes

4GB corpus:
~4,000,000,000 bytes

Conclusion:
4GB cannot be treated as a normal continuation of the current build_sa path.

## Additional Finding

src/build_sa_u32.cpp exists.

It appears to:
- use int64_t internally
- validate uint32_t output range
- write uint32_t SA

This may support >2GB and <4GB corpus positions, but it is still not SA64.

It should be treated as SA32/u32 boundary path, not full SA64.

## Risk

Running 4GB latency scaling without explicit SA boundary validation risks:

- overflow
- signed/unsigned narrowing
- silent SA corruption
- invalid BWT
- invalid FM index
- false latency physics conclusions

Silent corruption is worse than a hard failure.

## Current Rule

Do not run 4GB latency scaling as a physics measurement until one of these is true:

1. SA64 path is implemented and validated.
2. build_sa_u32 path is explicitly validated for 4GB-safe operation.
3. 4GB corpus is split/sharded below SA32/int32 boundaries.

## Safe Next Steps

1. Create SA32 boundary tests.
2. Validate build_sa_u32 on >2GB synthetic corpus if feasible.
3. Decide whether GLYPH v0.x supports:
   - monolithic <=2GB safe path
   - u32 <=4GB boundary path
   - true SA64 path
   - segmented/sharded path

## Current Scaling Law Scope

LATENCY_SCALING_LAW_V1 is currently valid through 2GB corpus scale only.

Do not extend it to 4GB until SA boundary is resolved.

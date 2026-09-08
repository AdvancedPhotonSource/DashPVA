# Sample orientation

Two settings tell the software which way your goniometer turns:

- **Direction** — one per row in the circle tables
- **Sample orientation** — under **Advanced**

Get them wrong and you get no error, just a flipped or rotated map.

---

## 1. Order the circles

Top row = outermost, bottom row = innermost. Relative to the **sample** for sample
circles, the **detector** for detector circles.

- **Outermost** — furthest from it, mounted to the base. Move it, everything moves.
- **Innermost** — closest to it. Move it, only the sample (or detector) turns.

Reorder with **Move up** / **Move down**. Renaming does nothing.

Only the bottom row feeds the orientation setting: `det` reads the bottom **detector**
circle, `sam` the bottom **sample** circle. The rows above it take no part in that
decision — so reordering the table changes the orientation, even with the same
circles listed.

---

## 2. Set each Direction

The axis the circle turns **around**, plus a sign:

| Value | Where |
|---|---|
| `x+` `x-` `y+` `y-` `z+` `z-` | sample or detector |
| `k+` `k-` | sample only (kappa) |

Degrees only.

The sign is the direction the axis vector points in xrayutilities' convention — not
necessarily the motor's positive direction. Confirm it; don't guess.

---

## 3. Choose a sample orientation

| Value | Meaning | Requires |
|---|---|---|
| `det` | from the detector circles | ≥1 detector circle; bottom one not along the beam (falls back to the one above) |
| `sam` | from the sample circles | ≥1 sample circle; bottom one not along the beam |
| `x+` `z-` … | stated directly | not along the beam |

`sam` is only correct if your bottom sample circle is the **azimuth motor**. Anything
else and it's wrong even though it's accepted — hence the warning.

---

## Checked vs not

Rejected: invalid value; `det`/`sam` with no matching circles; deciding axis along the
beam.

**Not** checked: circle order, direction signs, and whether your beam / in-plane /
surface-normal vectors share a frame with the circles. These fail silently.

---

## Example: APS 6ID

The shipped default. Illustration, not values to copy.

| Position | Type | Motor | Label | Direction |
|---|---|---|---|---|
| outermost | sample | `6idb1:m28.RBV` | Mu | `x+` |
| | sample | `6idb1:m17.RBV` | Eta | `z-` |
| | sample | `6idb1:m19.RBV` | Chi | `y+` |
| innermost | sample | `6idb1:m20.RBV` | Phi | `z-` |
| outermost | detector | `6idb1:m29.RBV` | Nu | `x+` |
| innermost | detector | `6idb1:m18.RBV` | Delta | `z-` |

| Setting | Value |
|---|---|
| Sample orientation | `det` |
| Beam direction | `0, 1, 0` |
| In-plane reference | `0, 1, 0` |
| Surface normal | `0, 0, 1` |

Note the directions are mixed with no pattern, `z-` repeats across two motors, and
Delta (`z-`) isn't along the beam (`0, 1, 0`) — which is what makes `det` valid here.

---

## Your setup

Live values are in this window, not this page. To document a beamline, record each
circle's position, motor, and direction, plus the orientation and the three vectors.

Settle one thing with the instrument scientist: which physical way each motor turns,
and how that becomes `+` or `-`. Mirrored map? Check that first.

---

## Once configured: it becomes an IOC

Everything set here is served as live EPICS records under your **IOC prefix**. The
profile is the definition; the running IOC is how the rest of the beamline reads it.

With prefix `6idb:`, the six circles above become 39 records:

| Pattern | Example | Holds |
|---|---|---|
| `{prefix}{circle}:Position` | `6idb:Phi:Position` | live angle |
| `{prefix}{circle}:DirectionAxis` | `6idb:Phi:DirectionAxis` | its direction (`z-`) |
| `{prefix}{circle}:AxisNumber` | `6idb:Phi:AxisNumber` | its place in the stack |
| `{prefix}spec:Energy:Value` | `6idb:spec:Energy:Value` | energy |
| `{prefix}spec:UB_matrix:Value` | `6idb:spec:UB_matrix:Value` | UB matrix |
| `{prefix}DetectorSetup:*` | `6idb:DetectorSetup:Distance` | detector geometry |
| `{prefix}ScanOn:Value` | `6idb:ScanOn:Value` | scan flag |
| `{prefix}FilePath:Value` `{prefix}FileName:Value` | | output file |

So `caget 6idb:Phi:DirectionAxis` returns what you typed in the Direction column, and
anything speaking Channel Access — SPEC, the HPC consumers, the viewers — reads the
geometry from these rather than from the profile file.

The **Live IOC records** table at the bottom of this window lists every record and its
current value. Double-click to copy a name.

Two things to know:

- **Apply & Save restarts the IOC** from the saved snapshot. Records briefly drop.
- **A `caput` to one of these is noticed.** Next time you save, a record changed on
  the IOC but untouched here is adopted into the profile; changed in both places to
  different values, you get a conflict and nothing is written.

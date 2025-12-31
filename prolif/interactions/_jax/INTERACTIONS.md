# ProLIF Interactions - JAX Implementation Status

## Summary

| Interaction | Type | JAX Status | Notes |
|-------------|------|------------|-------|
| Hydrophobic | Distance | ✅ | distance <= 4.5 |
| Cationic | Distance | ❌ | ligand cation + residue anion |
| Anionic | Distance | ❌ | ligand anion + residue cation (inverted Cationic) |
| VdWContact | Distance + Radii | ❌ | distance <= sum_vdw + tolerance |
| HBAcceptor | SingleAngle | ❌ | ligand acceptor + residue donor |
| HBDonor | SingleAngle | ❌ | ligand donor + residue acceptor (inverted HBAcceptor) |
| XBAcceptor | DoubleAngle | ❌ | ligand acceptor + residue halogen donor |
| XBDonor | DoubleAngle | ❌ | ligand halogen donor + residue acceptor (inverted XBAcceptor) |
| CationPi | Ring + Angle | ❌ | ligand cation + residue aromatic ring |
| PiCation | Ring + Angle | ❌ | ligand aromatic ring + residue cation (inverted CationPi) |
| FaceToFace | Ring + Angles | ❌ | parallel pi-stacking |
| EdgeToFace | Ring + Angles | ❌ | perpendicular pi-stacking + intersect check |
| PiStacking | Composite | ❌ | FaceToFace OR EdgeToFace |
| MetalDonor | Distance | ❌ | ligand metal + residue chelated |
| MetalAcceptor | Distance | ❌ | ligand chelated + residue metal (inverted MetalDonor) |

---

## Interaction Details

### 1. Hydrophobic (Distance)
**Status: ✅ Implemented**

```
Check: distance <= 4.5 Å
Atoms: hydrophobic atoms on both ligand and residue
```

**ProLIF Logic:**
- SMARTS identifies hydrophobic atoms (C, S, Br, I, specific carbon patterns)
- Simple distance check between matched atoms

---

### 2. Cationic (Distance)
**Status: ❌ Not Implemented**

```
Check: distance <= 4.5 Å
Ligand: cation [+{1-},$([NX3&!$([NX3]-O)]-[C]=[NX3+])]
Residue: anion [-{1-},$(O=[C,S,P]-[O-])]
```

**ProLIF Logic:**
- SMARTS identifies cations on ligand, anions on residue
- Simple distance check

---

### 3. Anionic (Distance)
**Status: ❌ Not Implemented**

```
Check: distance <= 4.5 Å
Ligand: anion
Residue: cation
```

**ProLIF Logic:**
- Inverted Cationic (swap ligand/residue patterns)

---

### 4. VdWContact (Distance + Radii)
**Status: ❌ Not Implemented**

```
Check: distance <= (vdw_radius_1 + vdw_radius_2 + tolerance)
Default tolerance: 0.0
```

**ProLIF Logic:**
- For each atom pair, look up VdW radii by element symbol
- Compare distance to sum of radii + tolerance
- Presets: mdanalysis, rdkit, csd

---

### 5. HBAcceptor (SingleAngle)
**Status: ❌ Not Implemented**

```
Check:
  1. distance(Acceptor, Donor) <= 3.5 Å
  2. 130° <= DHA_angle <= 180°

Where DHA_angle = angle(Donor, Hydrogen, Acceptor)

Ligand: acceptor atom
Residue: donor-hydrogen pair
```

**ProLIF Logic:**
- SMARTS for acceptor: N, O, F patterns (not positively charged)
- SMARTS for donor: [O,S,#7;+0]-[H] or [Nv4+1]-[H]
- Distance from acceptor to donor (not hydrogen)
- Angle at hydrogen vertex

---

### 6. HBDonor (SingleAngle)
**Status: ❌ Not Implemented**

```
Same as HBAcceptor but inverted:
Ligand: donor-hydrogen pair
Residue: acceptor atom
```

---

### 7. XBAcceptor (DoubleAngle)
**Status: ❌ Not Implemented**

```
Check:
  1. distance(Acceptor, Halogen) <= 3.5 Å
  2. 130° <= AXD_angle <= 180°
  3. 80° <= XAR_angle <= 140°

Where:
  AXD_angle = angle(Acceptor, Halogen, Donor)
  XAR_angle = angle(Halogen, Acceptor, R)

Ligand: acceptor-R pair
Residue: donor-halogen pair
```

**ProLIF Logic:**
- Acceptor SMARTS: [#7,#8,P,S,Se,Te,a;!+{1-}]!#[*]
- Donor SMARTS: [#6,#7,Si,F,Cl,Br,I]-[Cl,Br,I,At]
- Two angle checks required

---

### 8. XBDonor (DoubleAngle)
**Status: ❌ Not Implemented**

```
Same as XBAcceptor but inverted:
Ligand: donor-halogen pair
Residue: acceptor-R pair
```

---

### 9. CationPi (Ring + Angle)
**Status: ❌ Not Implemented**

```
Check:
  1. distance(Cation, Ring_Centroid) <= 4.5 Å
  2. 0° <= angle <= 30° (or 150° <= angle <= 180°)

Where angle = angle between ring normal and centroid→cation vector

Ligand: cation
Residue: aromatic ring (5 or 6 membered)
```

**ProLIF Logic:**
- Cation SMARTS: [+{1-},$([NX3&!$([NX3]-O)]-[C]=[NX3+])]
- Ring SMARTS: [a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1 (6-membered)
             [a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1 (5-membered)
- Compute ring centroid
- Compute ring normal vector
- Check angle (cation can be above or below ring)

---

### 10. PiCation (Ring + Angle)
**Status: ❌ Not Implemented**

```
Same as CationPi but inverted:
Ligand: aromatic ring
Residue: cation
```

---

### 11. FaceToFace (BasePiStacking)
**Status: ❌ Not Implemented**

```
Check:
  1. distance(Centroid1, Centroid2) <= 5.5 Å
  2. 0° <= plane_angle <= 35°
  3. 0° <= normal_to_centroid_angle <= 33°

Where:
  plane_angle = angle between ring planes (using normals)
  normal_to_centroid_angle = min angle from either ring's normal to centroid-centroid vector
```

**ProLIF Logic:**
- Parallel rings (small plane angle)
- Centroids roughly aligned with ring normals

---

### 12. EdgeToFace (BasePiStacking)
**Status: ❌ Not Implemented**

```
Check:
  1. distance(Centroid1, Centroid2) <= 6.5 Å
  2. 50° <= plane_angle <= 90°
  3. 0° <= normal_to_centroid_angle <= 30°
  4. Intersect point within intersect_radius (1.5 Å) of opposite centroid

Where intersect = point where perpendicular ring's plane intersects other ring's plane
```

**ProLIF Logic:**
- Perpendicular rings (large plane angle)
- Additional geometric check for intersection point

---

### 13. PiStacking (Composite)
**Status: ❌ Not Implemented**

```
Check: FaceToFace OR EdgeToFace
```

**ProLIF Logic:**
- Simply calls both FaceToFace.detect() and EdgeToFace.detect()
- Yields results from both

---

### 14. MetalDonor (Distance)
**Status: ❌ Not Implemented**

```
Check: distance <= 2.8 Å
Ligand: metal [Ca,Cd,Co,Cu,Fe,Mg,Mn,Ni,Zn]
Residue: chelating atom [O,#7...;!+{1-}]
```

**ProLIF Logic:**
- Simple distance check between metal and chelating atoms

---

### 15. MetalAcceptor (Distance)
**Status: ❌ Not Implemented**

```
Same as MetalDonor but inverted:
Ligand: chelating atom
Residue: metal
```

---

## Base Classes

### Distance
Simple distance check between ligand pattern and residue pattern.

### SingleAngle
Distance check + one angle check (3 atoms: L1, P1, P2).

### DoubleAngle
Distance check + two angle checks (4 atoms: L1, L2, P1, P2).

### BasePiStacking
Ring-based interaction with centroid distance, plane angle, and normal-to-centroid angle checks.

### Interaction
Base class, custom detect() method.

---

## JAX Implementation Notes

1. **Distance-based** (Hydrophobic, Cationic, Anionic, MetalDonor, MetalAcceptor):
   - Use `pairwise_distances()` primitive
   - Simple threshold comparison

2. **SingleAngle** (HBAcceptor, HBDonor):
   - Use `pairwise_distances()` + `angle_at_vertex()`
   - Need to identify donor-hydrogen pairs

3. **DoubleAngle** (XBAcceptor, XBDonor):
   - Use `pairwise_distances()` + two `angle_at_vertex()` calls
   - Need 4 atom positions

4. **Ring-based** (CationPi, PiCation, FaceToFace, EdgeToFace):
   - Use `batch_centroids()`, `batch_ring_normals()`
   - Use `angle_between_vectors()` for normal/vector angles
   - EdgeToFace needs additional intersect point calculation

5. **VdWContact**:
   - Use `pairwise_distances()`
   - Need element-to-radius lookup
   - Compare to sum of radii per pair

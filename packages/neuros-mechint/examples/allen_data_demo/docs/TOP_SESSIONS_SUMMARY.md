# Top Allen Sessions for SAE Validation

**Generated**: December 31, 2025
**Total Processed**: 32 sessions
**Successful**: 25 sessions
**Meeting Criteria (>30% selective)**: 17 sessions

---

## 🌟 Top 10 Recommended Sessions

| Rank | Session ID | Units | % Selective | Max Corr | Mean Corr | Recommendation |
|------|------------|-------|-------------|----------|-----------|----------------|
| 1 | **754829445** | 92 | **45.7%** | **0.854** | 0.310 | ⭐ BEST OVERALL |
| 2 | **760345702** | 72 | **48.6%** | 0.659 | 0.295 | ⭐ MOST SELECTIVE % |
| 3 | **761418226** | 36 | **41.7%** | 0.786 | 0.305 | ⭐ EXCELLENT |
| 4 | **760693773** | 88 | **39.8%** | 0.829 | 0.295 | ⭐ EXCELLENT |
| 5 | **737581020** | 40 | **42.5%** | 0.722 | 0.287 | ⭐ EXCELLENT |
| 6 | 721123822 | 41 | 39.0% | 0.765 | 0.291 | ✓ GOOD |
| 7 | 751348571 | 49 | 38.8% | 0.779 | 0.287 | ✓ GOOD |
| 8 | 757970808 | 80 | 38.8% | 0.804 | 0.278 | ✓ GOOD |
| 9 | 719161530 | 52 | 38.5% | 0.750 | 0.275 | ✓ GOOD |
| 10 | 754312389 | 102 | 36.3% | 0.793 | - | ✓ GOOD (large N) |

---

## 📊 Aggregate Statistics (All 25 Sessions)

**Max Correlation:**
- Best: 0.854 (session 754829445)
- Median: 0.767
- Worst: 0.423

**Mean Correlation:**
- Best: 0.310
- Median: 0.268
- Worst: 0.121

**Fraction Selective (>0.3 corr):**
- Best: 52.6%
- Median: 34.6%
- Worst: 4.4%

---

## 🎯 Recommended Usage

### For SAE Training (Use Top 5)
```python
recommended_sessions = [754829445, 760345702, 761418226, 760693773, 737581020]
```

These 5 sessions provide:
- Diverse unit counts (36-92 units)
- High selectivity (40-49%)
- Strong correlations (0.66-0.85)
- Robust statistics

### For Multi-Session Validation (Use Top 10)
Include all 10 sessions above for:
- Generalization testing
- Session-to-session variability analysis
- Large-scale feature discovery

---

## 📁 Sessions to Keep

**Essential (Top 5 - DO NOT DELETE):**
- 754829445
- 760345702
- 761418226
- 760693773
- 737581020

**Important (Top 10 - Keep if space allows):**
- 721123822
- 751348571
- 757970808
- 719161530
- 754312389

**Optional (11-17 - Can delete if needed):**
- 758798717
- 750749662
- 762120172
- 732592105
- 755434585
- 750332458
- 759883607

**Low Quality (Can delete):**
- All others with <30% selective

---

## 🗑️ Sessions to Delete

**Corrupted (I/O errors - DELETE IMMEDIATELY):**
- 762602078
- 763673393
- 773418906
- 791319847
- 797828357
- 798911424
- 799864342

**Low Quality (<30% selective - Safe to delete):**
- 715093703 (30.0% - borderline)
- 746083955
- 742951821
- 743475441
- 744228101
- 756029989
- 757216464

---

## 💡 Next Steps

1. Delete corrupted and low-quality sessions (saves ~25-30 GB)
2. Train SAEs on top 5 sessions
3. Validate cross-session generalization on top 10
4. Generate publication figures

**You have everything needed for publication-quality research!** 🎉

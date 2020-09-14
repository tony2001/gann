package gann

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/mathetake/gann/metric"
)

func TestIndex_GetANNbyItemID(t *testing.T) {
	for i, c := range []struct {
		dim, num, nTree, k int
	}{
		{dim: 2, num: 1000, nTree: 10, k: 2},
		{dim: 10, num: 100, nTree: 5, k: 10},
		{dim: 1000, num: 10000, nTree: 5, k: 10},
	} {
		c := c
		t.Run(fmt.Sprintf("%d-th case", i), func(t *testing.T) {
			rawItems := make([][]float64, c.num)
			for i := range rawItems {
				v := make([]float64, c.dim)

				var norm float64
				for j := range v {
					cof := rand.Float64() - 0.5
					v[j] = cof
					norm += cof * cof
				}

				norm = math.Sqrt(norm)
				for j := range v {
					v[j] /= norm
				}

				rawItems[i] = v
			}

			m, err := metric.NewCosineMetric(c.dim)
			if err != nil {
				t.Fatal(err)
			}

			idx, err := CreateNewIndex(rawItems, c.dim, c.nTree, c.k, m)
			if err != nil {
				t.Fatal(err)
			}

			if _, err = idx.GetANNbyItemID(0, 10, 2); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestIndex_GetANNbyVector(t *testing.T) {
	for i, c := range []struct {
		dim, num, nTree, k int
	}{
		{dim: 2, num: 1000, nTree: 10, k: 2},
		{dim: 10, num: 100, nTree: 5, k: 10},
		{dim: 1000, num: 10000, nTree: 5, k: 10},
	} {
		c := c
		t.Run(fmt.Sprintf("%d-th case", i), func(t *testing.T) {
			rawItems := make([][]float64, c.num)
			for i := range rawItems {
				v := make([]float64, c.dim)

				var norm float64
				for j := range v {
					cof := rand.Float64() - 0.5
					v[j] = cof
					norm += cof * cof
				}

				norm = math.Sqrt(norm)
				for j := range v {
					v[j] /= norm
				}

				rawItems[i] = v
			}

			m, err := metric.NewCosineMetric(c.dim)
			if err != nil {
				t.Fatal(err)
			}

			idx, err := CreateNewIndex(rawItems, c.dim, c.nTree, c.k, m)
			if err != nil {
				t.Fatal(err)
			}

			key := make([]float64, c.dim)
			for i := range key {
				key[i] = rand.Float64() - 0.5
			}

			if _, err = idx.GetANNbyVector(key, 10, 2); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// This unit test is made to verify if our algorithm can correctly find
// the `exact` neighbors. That is done by checking the ratio of exact
// neighbors in the result returned by `getANNbyVector` is less than
// the given threshold.
func TestAnnSearchAccuracy(t *testing.T) {
	for i, c := range []struct {
		k, dim, num, nTree, searchNum int
		threshold, bucketScale        float64
	}{
		{
			k:           2,
			dim:         20,
			num:         10000,
			nTree:       20,
			threshold:   0.90,
			searchNum:   200,
			bucketScale: 20,
		},
		{
			k:           2,
			dim:         20,
			num:         10000,
			nTree:       20,
			threshold:   0.8,
			searchNum:   20,
			bucketScale: 1000,
		},
	} {
		c := c
		t.Run(fmt.Sprintf("%d-th case", i), func(t *testing.T) {
			rawItems := make([][]float64, c.num)
			for i := range rawItems {
				v := make([]float64, c.dim)

				var norm float64
				for j := range v {
					cof := rand.Float64() - 0.5
					v[j] = cof
					norm += cof * cof
				}

				norm = math.Sqrt(norm)
				for j := range v {
					v[j] /= norm
				}

				rawItems[i] = v
			}

			m, err := metric.NewCosineMetric(c.dim)
			if err != nil {
				t.Fatal(err)
			}

			idx, err := CreateNewIndex(rawItems, c.dim, c.nTree, c.k, m)
			if err != nil {
				t.Fatal(err)
			}

			rawIdx, ok := idx.(*index)
			if !ok {
				t.Fatal("assertion failed")
			}

			// query vector
			query := make([]float64, c.dim)
			query[0] = 0.1

			// exact neighbors
			aDist := map[int64]float64{}
			ids := make([]int64, len(rawItems))
			for i, v := range rawItems {
				ids[i] = int64(i)
				aDist[int64(i)] = rawIdx.metric.CalcDistance(v, query)
			}
			sort.Slice(ids, func(i, j int) bool {
				return aDist[ids[i]] < aDist[ids[j]]
			})

			expectedIDsMap := make(map[int64]struct{}, c.searchNum)
			for _, id := range ids[:c.searchNum] {
				expectedIDsMap[int64(id)] = struct{}{}
			}

			ass, err := idx.GetANNbyVector(query, c.searchNum, c.bucketScale)
			if err != nil {
				t.Fatal(err)
			}

			var count int
			for _, id := range ass {
				if _, ok := expectedIDsMap[id]; ok {
					count++
				}
			}

			if ratio := float64(count) / float64(c.searchNum); ratio < c.threshold {
				t.Fatalf("Too few exact neighbors found in approximated result: %d / %d = %f", count, c.searchNum, ratio)
			} else {
				t.Logf("ratio of exact neighbors in approximated result: %d / %d = %f", count, c.searchNum, ratio)
			}
		})
	}
}

func hashDistance(hash1 []float64, hash2 []float64) float64 {
	distance := float64(0)
	for i := 0; i < len(hash1); i++ {
		tmp := (hash1[i] - hash2[i]) * 1000
		distance += float64(tmp * tmp)
	}
	return distance
}

func CopyHash(hash []float64) []float64 {
	copy := make([]float64, len(hash))
	for i := 0; i < len(hash); i++ {
		copy[i] = hash[i]
	}
	return copy
}

func getRandomInt8() float64 {
	return float64((rand.Intn(128) - 128)) / 1000
}

func TestIndex_GetANNbyVector_Existing(t *testing.T) {
	dim := 144
	num := 100000
	nTree := 4
	k := 2

	rawItems := make([][]float64, num)
	for i := range rawItems {
		v := make([]float64, dim)

		var norm float64
		for j := range v {
			cof := getRandomInt8()
			v[j] = cof
			norm += cof * cof
		}

		norm = math.Sqrt(norm)
		for j := range v {
			v[j] /= norm
		}
		rawItems[i] = v
	}

	m, err := metric.NewCosineMetric(dim)
	if err != nil {
		t.Fatal(err)
	}

	idx, err := CreateNewIndex(rawItems, dim, nTree, k, m)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < num; i++ {
		keyCopy := CopyHash(rawItems[i])
		res, _ := idx.GetANNbyVector(keyCopy, 20, 6)

		if int64(i) != res[0] {
			t.Fatalf("%d not the first\n", i)
		}
	}
}

func testIndex_GetANNbyVector_Existing_With_Diff(t *testing.T, diffInt int, num int) {
	dim := 144
	nTree := 4
	k := 2

	rawItems := make([][]float64, num)
	for i := range rawItems {
		v := make([]float64, dim)

		var norm float64
		for j := range v {
			cof := getRandomInt8()
			v[j] = cof
			norm += cof * cof
		}
		norm = math.Sqrt(norm)
		for j := range v {
			v[j] /= norm
		}
		rawItems[i] = v
	}

	m, err := metric.NewCosineMetric(dim)
	if err != nil {
		t.Fatal(err)
	}

	idx, err := CreateNewIndex(rawItems, dim, nTree, k, m)
	if err != nil {
		t.Fatal(err)
	}

	diff := float64(diffInt) / 1000

	for i := 0; i < num; i++ {
		keyCopy := CopyHash(rawItems[i])
		for j := 0; j < len(keyCopy); j++ {
			if 255-keyCopy[j] > diff {
				keyCopy[j] += diff
			} else {
				keyCopy[j] -= diff
			}
		}

		res, _ := idx.GetANNbyVector(keyCopy, 20, 6)

		if int64(i) != res[0] {
			t.Fatalf("%d not the first\n", i)
		}
	}
}

func TestIndex_GetANNbyVector_Existing_With_Diff_1(t *testing.T) {
	testIndex_GetANNbyVector_Existing_With_Diff(t, 1, 20000)
}

func TestIndex_GetANNbyVector_Existing_With_Diff_2(t *testing.T) {
	testIndex_GetANNbyVector_Existing_With_Diff(t, 2, 20000)
}

func TestIndex_GetANNbyVector_Existing_With_Diff_5(t *testing.T) {
	testIndex_GetANNbyVector_Existing_With_Diff(t, 5, 20000)
}

func TestIndex_GetANNbyVector_Existing_With_Diff_10(t *testing.T) {
	testIndex_GetANNbyVector_Existing_With_Diff(t, 10, 20000)
}

func TestIndex_GetANNbyVector_Existing_With_Diff_20(t *testing.T) {
	testIndex_GetANNbyVector_Existing_With_Diff(t, 20, 20000)
}

func BenchmarkIndex_GetANNbyVector_Random_100000(b *testing.B) {
	dim := 144
	num := 100000
	nTree := 4
	k := 2

	rawItems := make([][]float64, num)
	for i := range rawItems {
		v := make([]float64, dim)

		var norm float64
		for j := range v {
			cof := getRandomInt8()
			v[j] = cof
			norm += cof * cof
		}

		norm = math.Sqrt(norm)
		for j := range v {
			v[j] /= norm
		}
		rawItems[i] = v
	}

	m, err := metric.NewCosineMetric(dim)
	if err != nil {
		b.Fatal(err)
	}

	idx, err := CreateNewIndex(rawItems, dim, nTree, k, m)
	if err != nil {
		b.Fatal(err)
	}

	key := make([]float64, dim)
	for i := range key {
		key[i] = getRandomInt8()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := idx.GetANNbyVector(key, 10, 2)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkIndex_GetANNbyVector_Existing_50000(b *testing.B) {
	dim := 144
	num := 50000
	nTree := 4
	k := 2

	rawItems := make([][]float64, num)
	for i := range rawItems {
		v := make([]float64, dim)

		var norm float64
		for j := range v {
			cof := getRandomInt8()
			v[j] = cof
			norm += cof * cof
		}

		rawItems[i] = v
	}

	m, err := metric.NewCosineMetric(dim)
	if err != nil {
		b.Fatal(err)
	}

	idx, err := CreateNewIndex(rawItems, dim, nTree, k, m)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		keyNum := i % len(rawItems)
		key := rawItems[keyNum]

		res, err := idx.GetANNbyVector(key, 10, 2)
		if err != nil {
			b.Fatal(err)
		}

		if len(res) == 0 {
			b.Fatalf("looking for %d, but found none", keyNum)
		}

		if res[0] != int64(keyNum) {
			inRes := false
			for _, foundId := range res {
				if foundId == int64(keyNum) {
					inRes = true
					break
				}
			}
			if !inRes {
				found, _ := idx.GetVectorByItemId(res[0])
				b.Fatalf("looking for %d:\n%v\n, but found %d (%v):\n%v\ndistance: %f\n", keyNum, rawItems[keyNum], res[0], res, found, hashDistance(rawItems[keyNum], found))
			}
		}
	}
}

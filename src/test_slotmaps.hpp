// test_slotmaps.cpp
// Compile example:
//   g++ -std=c++17 -Wall -Wextra -pedantic -I. test_slotmaps.cpp -o test_slotmaps
//
// If your local header names differ, either rename them to sparse_slotmap.hpp / dense_slotmap.hpp,
// or pass include overrides, for example:
//   g++ -std=c++17 -Wall -Wextra -pedantic -I. -DSPARSE_SLOTMAP_HEADER='"sparse_slotmap(2).hpp"'
//       -DDENSE_SLOTMAP_HEADER='"dense_slotmap(1).hpp"'
//       test_slotmaps.cpp -o test_slotmaps

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>      // Needed before the current slotmap_commons.hpp because it uses std::numeric_limits.
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef SPARSE_SLOTMAP_HEADER
    #if __has_include("sparse_slotmap(2).hpp")
        #define SPARSE_SLOTMAP_HEADER "sparse_slotmap(2).hpp"
    #elif __has_include("sparse_slotmap.hpp")
        #define SPARSE_SLOTMAP_HEADER "sparse_slotmap.hpp"
    #elif __has_include("sparse_slotmap(1).hpp")
        #define SPARSE_SLOTMAP_HEADER "sparse_slotmap(1).hpp"
    #else
        #error "Could not find sparse slot map header. Define SPARSE_SLOTMAP_HEADER."
    #endif
#endif

#ifndef DENSE_SLOTMAP_HEADER
    #if __has_include("dense_slotmap(1).hpp")
        #define DENSE_SLOTMAP_HEADER "dense_slotmap(1).hpp"
    #elif __has_include("dense_slotmap.hpp")
        #define DENSE_SLOTMAP_HEADER "dense_slotmap(1).hpp"
    #else
        #error "Could not find dense slot map header. Define DENSE_SLOTMAP_HEADER."
    #endif
#endif

#include SPARSE_SLOTMAP_HEADER
#include DENSE_SLOTMAP_HEADER

struct Payload
{
    int id = 0;
    std::string name;

    Payload() = default;
    Payload(int _id, std::string _name)
        : id(_id)
        , name(std::move(_name))
    {
    }
};

static std::string bool_string(bool value)
{
    return value ? "true" : "false";
}

template <typename T>
static std::string value_string(const T& value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

static std::string value_string(const std::string& value)
{
    return '"' + value + '"';
}

static std::string vector_string(std::vector<int> values)
{
    std::ostringstream oss;
    oss << '[';
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i != 0)
        {
            oss << ", ";
        }
        oss << values[i];
    }
    oss << ']';
    return oss.str();
}

class TestRunner
{
public:
    void section(const std::string& name)
    {
        std::cout << "\n=== " << name << " ===\n";
    }

    void check(const std::string& testName, bool passed, const std::string& expected, const std::string& actual)
    {
        ++totalChecks;
        if (!passed)
        {
            ++failedChecks;
        }

        std::cout << (passed ? "[PASS] " : "[FAIL] ") << testName
                  << " | expected: " << expected
                  << " | actual: " << actual << '\n';
    }

    void expect_true(const std::string& testName, bool actual)
    {
        check(testName, actual == true, "true", bool_string(actual));
    }

    void expect_false(const std::string& testName, bool actual)
    {
        check(testName, actual == false, "false", bool_string(actual));
    }

    template <typename T, typename U>
    void expect_eq(const std::string& testName, const T& expected, const U& actual)
    {
        check(testName, expected == actual, value_string(expected), value_string(actual));
    }

    template <typename T, typename U>
    void expect_ne(const std::string& testName, const T& notExpected, const U& actual)
    {
        check(testName, notExpected != actual, "not " + value_string(notExpected), value_string(actual));
    }

    void expect_vector_eq(const std::string& testName, std::vector<int> expected, std::vector<int> actual)
    {
        check(testName, expected == actual, vector_string(expected), vector_string(actual));
    }

    int failed() const
    {
        return failedChecks;
    }

    int total() const
    {
        return totalChecks;
    }

private:
    int totalChecks = 0;
    int failedChecks = 0;
};

template <typename Map>
static std::vector<int> sorted_ids_from_values(const Map& map)
{
    std::vector<int> ids;
    for (const Payload& value : map)
    {
        ids.push_back(value.id);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

template <typename Map>
static std::vector<int> sorted_ids_from_items(const Map& map, bool& allKeysResolved)
{
    std::vector<int> ids;
    allKeysResolved = true;

    for (const auto& kv : map.items())
    {
        const typename Map::key key = kv.first;
        const Payload& value = kv.second;
        const Payload* resolved = map.get(key);

        if (resolved == nullptr || resolved->id != value.id || resolved->name != value.name)
        {
            allKeysResolved = false;
        }

        ids.push_back(value.id);
    }

    std::sort(ids.begin(), ids.end());
    return ids;
}

template <typename Map>
static void run_common_slotmap_tests(TestRunner& runner, const std::string& mapName)
{
    using Key = typename Map::key;
    using RawKey = typename Key::id_type;

    runner.section(mapName + " common behavior");

    Map map;

    runner.expect_true(mapName + ": new map is empty", map.empty());
    runner.expect_eq(mapName + ": new map size", 0u, map.size());

    Key keyA = map.emplace(10, "A");
    Key keyB = map.emplace(20, "B");
    Key keyC = map.emplace(30, "C");

    runner.expect_false(mapName + ": map is not empty after emplace", map.empty());
    runner.expect_eq(mapName + ": size after 3 emplaces", 3u, map.size());
    runner.expect_true(mapName + ": contains A", map.contains(keyA));
    runner.expect_true(mapName + ": contains B", map.contains(keyB));
    runner.expect_true(mapName + ": contains C", map.contains(keyC));

    Payload* valueA = map.get(keyA);
    Payload* valueB = map.get(keyB);
    Payload* valueC = map.get(keyC);

    runner.expect_true(mapName + ": get A returns non-null", valueA != nullptr);
    runner.expect_true(mapName + ": get B returns non-null", valueB != nullptr);
    runner.expect_true(mapName + ": get C returns non-null", valueC != nullptr);

    runner.expect_eq(mapName + ": A id", 10, valueA ? valueA->id : -1);
    runner.expect_eq(mapName + ": B name", std::string("B"), valueB ? valueB->name : std::string("<null>"));
    runner.expect_eq(mapName + ": C id", 30, valueC ? valueC->id : -1);

    if (valueB != nullptr)
    {
        valueB->id = 25;
        valueB->name = "B2";
    }
    runner.expect_eq(mapName + ": mutable get can edit id", 25, map.get(keyB) ? map.get(keyB)->id : -1);
    runner.expect_eq(mapName + ": mutable get can edit name", std::string("B2"), map.get(keyB) ? map.get(keyB)->name : std::string("<null>"));

    map.erase(keyB);
    runner.expect_eq(mapName + ": size after erasing B", 2u, map.size());
    runner.expect_false(mapName + ": erased key B is not contained", map.contains(keyB));
    runner.expect_true(mapName + ": get erased key B returns null", map.get(keyB) == nullptr);
    runner.expect_true(mapName + ": A still valid after erasing B", map.contains(keyA));
    runner.expect_true(mapName + ": C still valid after erasing B", map.contains(keyC));

    Key keyD = map.emplace(40, "D");
    runner.expect_eq(mapName + ": size after emplacing D", 3u, map.size());
    runner.expect_true(mapName + ": D is contained", map.contains(keyD));
    runner.expect_true(mapName + ": old B key remains invalid after D reuse", map.get(keyB) == nullptr);
    runner.expect_eq(mapName + ": D id", 40, map.get(keyD) ? map.get(keyD)->id : -1);

    // All aliases below use MINFREEINDICES = 0, so the erased sparse index should be reused immediately.
    runner.expect_eq(mapName + ": erased key sparse index is reused", Key::toIndex(keyB), Key::toIndex(keyD));
    runner.expect_ne(mapName + ": reused key has newer version", Key::toVersion(keyB), Key::toVersion(keyD));
    runner.expect_ne(mapName + ": reused key has different raw value", static_cast<RawKey>(keyB), static_cast<RawKey>(keyD));

    runner.expect_vector_eq(mapName + ": value iteration sees alive ids", {10, 30, 40}, sorted_ids_from_values(map));

    bool itemKeysResolved = false;
    std::vector<int> itemIds = sorted_ids_from_items(map, itemKeysResolved);
    runner.expect_vector_eq(mapName + ": items() iteration sees alive ids", {10, 30, 40}, itemIds);
    runner.expect_true(mapName + ": every items() key resolves to its value", itemKeysResolved);

    std::optional<Payload> poppedD = map.pop(keyD);
    runner.expect_true(mapName + ": pop D returns a value", poppedD.has_value());
    runner.expect_eq(mapName + ": popped D id", 40, poppedD ? poppedD->id : -1);
    runner.expect_false(mapName + ": D is not contained after pop", map.contains(keyD));
    runner.expect_true(mapName + ": get D returns null after pop", map.get(keyD) == nullptr);
    runner.expect_eq(mapName + ": size after pop D", 2u, map.size());

    map.clear();
    runner.expect_true(mapName + ": clear makes map empty", map.empty());
    runner.expect_eq(mapName + ": size after clear", 0u, map.size());
    runner.expect_false(mapName + ": A invalid after clear", map.contains(keyA));
    runner.expect_false(mapName + ": C invalid after clear", map.contains(keyC));
    runner.expect_true(mapName + ": get A returns null after clear", map.get(keyA) == nullptr);

    Key keyE = map.emplace(50, "E");
    runner.expect_eq(mapName + ": can emplace after clear", 1u, map.size());
    runner.expect_eq(mapName + ": E id", 50, map.get(keyE) ? map.get(keyE)->id : -1);

    map.reset();
    runner.expect_true(mapName + ": reset makes map empty", map.empty());
    runner.expect_eq(mapName + ": size after reset", 0u, map.size());
    runner.expect_true(mapName + ": get E returns null immediately after reset", map.get(keyE) == nullptr);

    typename Map::Stats stats = map.debug_stats();
    runner.expect_eq(mapName + ": debug_stats alive count after reset", 0u, stats.numAliveItems);
}

static void run_sparse_pointer_stability_test(TestRunner& runner)
{
    using SparseMap = sparse_slotmap::slot_map<Payload, sparse_slotmap::slot_map_key64<Payload>, 8, 0>;

    runner.section("Sparse paged pointer stability behavior");

    SparseMap map;
    auto keyFirst = map.emplace(1, "first");
    Payload* firstPtr = map.get(keyFirst);
    const std::uintptr_t firstAddressBefore = reinterpret_cast<std::uintptr_t>(firstPtr);

    for (int i = 0; i < 64; ++i)
    {
        (void)map.emplace(1000 + i, "extra");
    }

    const std::uintptr_t firstAddressAfterGrowth = reinterpret_cast<std::uintptr_t>(map.get(keyFirst));
    runner.expect_eq("Sparse: pointer to existing value survives additional emplaces", firstAddressBefore, firstAddressAfterGrowth);

    auto keyOther = map.emplace(2, "other");
    Payload* otherPtr = map.get(keyOther);
    const std::uintptr_t otherAddressBefore = reinterpret_cast<std::uintptr_t>(otherPtr);

    map.erase(keyFirst);

    const std::uintptr_t otherAddressAfterErase = reinterpret_cast<std::uintptr_t>(map.get(keyOther));
    runner.expect_eq("Sparse: pointer to unrelated live value survives erase", otherAddressBefore, otherAddressAfterErase);
}

static void run_dense_compaction_test(TestRunner& runner)
{
    using DenseMap = dense_slotmap::slot_map<Payload, dense_slotmap::slot_map_key64<Payload>, 0>;

    runner.section("Dense packed compaction behavior");

    DenseMap map;
    map.reserve(8, 8);

    auto keyA = map.emplace(1, "A");
    auto keyB = map.emplace(2, "B");
    auto keyC = map.emplace(3, "C");

    const std::uintptr_t cAddressBefore = reinterpret_cast<std::uintptr_t>(map.get(keyC));

    map.erase(keyB); // swap-removes C into B's dense slot.

    const std::uintptr_t cAddressAfter = reinterpret_cast<std::uintptr_t>(map.get(keyC));

    runner.expect_true("Dense: key C remains valid after B erase", map.contains(keyC));
    runner.expect_eq("Dense: key C still resolves to id 3", 3, map.get(keyC) ? map.get(keyC)->id : -1);
    runner.expect_ne("Dense: C address changes when compacted into erased slot", cAddressBefore, cAddressAfter);
    runner.expect_vector_eq("Dense: live ids remain densely iterable after erase", {1, 3}, sorted_ids_from_values(map));
    runner.expect_true("Dense: key A remains valid after B erase", map.contains(keyA));
}

void run_tests()
{
    using Sparse64 = sparse_slotmap::slot_map<Payload, sparse_slotmap::slot_map_key64<Payload>, 8, 0>;
    using Dense64 = dense_slotmap::slot_map<Payload, dense_slotmap::slot_map_key64<Payload>, 0>;
    using Sparse32 = sparse_slotmap::slot_map<Payload, sparse_slotmap::slot_map_key32<Payload>, 8, 0>;
    using Dense32 = dense_slotmap::slot_map<Payload, dense_slotmap::slot_map_key32<Payload>, 0>;

    static_assert(!std::is_same<typename Sparse64::key, typename Dense64::key>::value,
        "Sparse and dense key types should remain distinct even if they share common key implementation.");
    static_assert(!std::is_same<typename Sparse32::key, typename Dense32::key>::value,
        "Sparse and dense 32-bit key types should remain distinct.");

    TestRunner runner;

    runner.section("Compile-time key domain checks");
    runner.expect_true("Sparse64 and Dense64 key types are distinct", !std::is_same<typename Sparse64::key, typename Dense64::key>::value);
    runner.expect_true("Sparse32 and Dense32 key types are distinct", !std::is_same<typename Sparse32::key, typename Dense32::key>::value);

    run_common_slotmap_tests<Sparse64>(runner, "Sparse64");
    run_common_slotmap_tests<Dense64>(runner, "Dense64");
    run_common_slotmap_tests<Sparse32>(runner, "Sparse32");
    run_common_slotmap_tests<Dense32>(runner, "Dense32");

    run_sparse_pointer_stability_test(runner);
    run_dense_compaction_test(runner);

    std::cout << "\n=== Summary ===\n";
    std::cout << "Total checks: " << runner.total() << '\n';
    std::cout << "Failed checks: " << runner.failed() << '\n';

    if (runner.failed() == 0)
    {
        std::cout << "RESULT: PASS\n";
    }
    std::cout << "RESULT: FAIL\n";
}

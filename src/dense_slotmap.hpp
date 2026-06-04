#pragma once

// Dense, non-paged generational slot map.
// Similar public shape to sparse_slotmap::slot_map, but stores live values packed in a dense vector.
//
// Important semantic difference from the paged sparse slot map:
// - Keys/handles remain stable until erased/invalidated.
// - Pointers/references/iterators to values are NOT stable across emplace/erase because values live in std::vector
//   and erase uses swap-remove to keep the array dense.

#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

// You could override memory allocator by defining SLOT_MAP_ALLOC/SLOT_MAP_FREE macroses.
#if !defined(SLOT_MAP_ALLOC) || !defined(SLOT_MAP_FREE)

#if defined(_WIN32)
#include <xmmintrin.h>
#define SLOT_MAP_ALLOC(sizeInBytes, alignment) _mm_malloc(sizeInBytes, alignment)
#define SLOT_MAP_FREE(ptr) _mm_free(ptr)
#else
#include <stdlib.h>
#define SLOT_MAP_ALLOC(sizeInBytes, alignment) aligned_alloc(alignment, sizeInBytes)
#define SLOT_MAP_FREE(ptr) free(ptr)
#endif

#endif

// You could override asserts by defining SLOT_MAP_ASSERT macro.
#if !defined(SLOT_MAP_ASSERT)
#define SLOT_MAP_ASSERT(expression) assert(expression)
#endif

namespace stl
{
    // STL-compatible aligned allocator.
    // Note: some platforms do not support alignments smaller than alignof(void*).
    template <class T, size_t Alignment = std::max(alignof(T), alignof(void*))>
    struct Allocator
    {
    public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;
        using size_type = size_t;
        using difference_type = ptrdiff_t;

        template <class U>
        struct rebind
        {
            using other = Allocator<U, Alignment>;
        };

        Allocator() noexcept = default;
        Allocator(const Allocator&) noexcept = default;

        template <typename U>
        Allocator(const Allocator<U, Alignment>&) noexcept {}

        [[nodiscard]] pointer address(reference x) const noexcept { return &x; }
        [[nodiscard]] const_pointer address(const_reference x) const noexcept { return &x; }

        [[nodiscard]] pointer allocate(size_type n, [[maybe_unused]] const void* hint = nullptr)
        {
            size_t alignment = std::max(Alignment, alignof(void*));
            size_t numBytes = sizeof(value_type) * n;
            numBytes = (numBytes + (alignment - 1)) & ~(alignment - 1);
            pointer p = reinterpret_cast<pointer>(SLOT_MAP_ALLOC(numBytes, alignment));
            SLOT_MAP_ASSERT(p);
            return p;
        }

        void deallocate(pointer p, size_type) noexcept { SLOT_MAP_FREE(p); }

        [[nodiscard]] size_type max_size() const noexcept
        {
            return std::numeric_limits<size_type>::max() / sizeof(value_type);
        }

        template <class U, class... Args>
        void construct(U* p, Args&&... args)
        {
            new (reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);
        }

        template <class U>
        void destroy(U* p)
        {
            p->~U();
        }
    };

    template <class T1, class T2, size_t Alignment>
    bool operator==(const Allocator<T1, Alignment>&, const Allocator<T2, Alignment>&) noexcept
    {
        return true;
    }

    template <class T1, class T2, size_t Alignment>
    bool operator!=(const Allocator<T1, Alignment>&, const Allocator<T2, Alignment>&) noexcept
    {
        return false;
    }
} // namespace stl

namespace dense_slotmap
{
    /*
    Strongly typed 64-bit key.

    | Component | Number of bits          |
    |-----------|--------------------------|
    | version   | 32 (0..4,294,967,295)    |
    | index     | 32 (0..4,294,967,295)    |
    */
    template <typename T>
    struct slot_map_key64
    {
        using id_type = uint64_t;
        using version_t = uint32_t;
        using index_t = uint32_t;

        static inline constexpr version_t kInvalidVersion = 0x0u;
        static inline constexpr version_t kMinVersion = 0x1u;
        static inline constexpr version_t kMaxVersion = 0xffffffffu;
        static inline constexpr index_t kMaxIndex = 0xffffffffu;

        static inline constexpr id_type kIndexMask = 0x00000000ffffffffull;
        static inline constexpr id_type kVersionMask = 0xffffffff00000000ull;
        static inline constexpr id_type kVersionShift = 32ull;

        [[nodiscard]] static inline constexpr slot_map_key64 make(version_t version, index_t index) noexcept
        {
            SLOT_MAP_ASSERT(version != kInvalidVersion);
            SLOT_MAP_ASSERT(index <= kMaxIndex);

            id_type v = (static_cast<id_type>(version) << kVersionShift) & kVersionMask;
            id_type i = static_cast<id_type>(index) & kIndexMask;
            return slot_map_key64{ v | i };
        }

        [[nodiscard]] inline size_t hash() const noexcept { return std::hash<id_type>{}(raw); }

        [[nodiscard]] static inline slot_map_key64 updateVersion(slot_map_key64 k, version_t version) noexcept
        {
            SLOT_MAP_ASSERT(version != kInvalidVersion);
            id_type v = (static_cast<id_type>(version) << kVersionShift) & kVersionMask;
            return slot_map_key64{ (k.raw & ~kVersionMask) | v };
        }

        [[nodiscard]] static inline index_t toIndex(slot_map_key64 k) noexcept
        {
            return static_cast<index_t>(k.raw & kIndexMask);
        }

        [[nodiscard]] static inline version_t toVersion(slot_map_key64 k) noexcept
        {
            return static_cast<version_t>((k.raw & kVersionMask) >> kVersionShift);
        }

        [[nodiscard]] static inline version_t increaseVersion(version_t version) noexcept { return version + 1u; }

        slot_map_key64() noexcept = default;

        explicit slot_map_key64(id_type _raw) noexcept
            : raw(_raw)
        {
        }

        bool operator==(const slot_map_key64& other) const noexcept { return raw == other.raw; }
        bool operator!=(const slot_map_key64& other) const noexcept { return raw != other.raw; }
        bool operator<(const slot_map_key64& other) const noexcept { return raw < other.raw; }

        [[nodiscard]] explicit operator id_type() const noexcept { return raw; }

        [[nodiscard]] static inline slot_map_key64 invalid() noexcept { return slot_map_key64{ 0 }; }

        id_type raw = 0;
    };

    /*
    Strongly typed 32-bit key.

    | Component | Number of bits     |
    |-----------|---------------------|
    | version   | 12 (0..4095)        |
    | index     | 20 (0..1,048,575)   |
    */
    template <typename T>
    struct slot_map_key32
    {
        using id_type = uint32_t;
        using version_t = uint16_t;
        using index_t = uint32_t;

        static inline constexpr version_t kInvalidVersion = 0x0u;
        static inline constexpr version_t kMinVersion = 0x1u;
        static inline constexpr version_t kMaxVersion = 0x0fffu;
        static inline constexpr index_t kMaxIndex = 0x000fffffu;

        static inline constexpr id_type kIndexMask = 0x000fffffu;
        static inline constexpr id_type kVersionMask = 0xfff00000u;
        static inline constexpr id_type kVersionShift = 20u;

        [[nodiscard]] static inline constexpr slot_map_key32 make(version_t version, index_t index) noexcept
        {
            SLOT_MAP_ASSERT(version != kInvalidVersion);
            SLOT_MAP_ASSERT(version <= kMaxVersion);
            SLOT_MAP_ASSERT(index <= kMaxIndex);

            id_type v = (static_cast<id_type>(version) << kVersionShift) & kVersionMask;
            id_type i = static_cast<id_type>(index) & kIndexMask;
            return slot_map_key32{ v | i };
        }

        [[nodiscard]] inline size_t hash() const noexcept { return std::hash<id_type>{}(raw); }

        [[nodiscard]] static inline slot_map_key32 updateVersion(slot_map_key32 k, version_t version) noexcept
        {
            SLOT_MAP_ASSERT(version != kInvalidVersion);
            SLOT_MAP_ASSERT(version <= kMaxVersion);
            id_type v = (static_cast<id_type>(version) << kVersionShift) & kVersionMask;
            return slot_map_key32{ (k.raw & ~kVersionMask) | v };
        }

        [[nodiscard]] static inline index_t toIndex(slot_map_key32 k) noexcept
        {
            return static_cast<index_t>(k.raw & kIndexMask);
        }

        [[nodiscard]] static inline version_t toVersion(slot_map_key32 k) noexcept
        {
            return static_cast<version_t>((k.raw & kVersionMask) >> kVersionShift);
        }

        [[nodiscard]] static inline version_t increaseVersion(version_t version) noexcept
        {
            return static_cast<version_t>(version + 1u);
        }

        slot_map_key32() noexcept = default;

        explicit slot_map_key32(id_type _raw) noexcept
            : raw(_raw)
        {
        }

        bool operator==(const slot_map_key32& other) const noexcept { return raw == other.raw; }
        bool operator!=(const slot_map_key32& other) const noexcept { return raw != other.raw; }
        bool operator<(const slot_map_key32& other) const noexcept { return raw < other.raw; }

        [[nodiscard]] explicit operator id_type() const noexcept { return raw; }

        [[nodiscard]] static inline slot_map_key32 invalid() noexcept { return slot_map_key32{ 0 }; }

        id_type raw = 0;
    };

    template <typename T, typename TKeyType = slot_map_key64<T>, size_t MINFREEINDICES = 64>
    class slot_map
    {
    public:
        using key = TKeyType;
        using version_t = typename TKeyType::version_t;
        using index_t = typename TKeyType::index_t;
        using size_type = uint32_t;

        static inline constexpr size_type kMinFreeIndices = static_cast<size_type>(MINFREEINDICES);

    private:
        struct SparseSlot
        {
            version_t version = key::kMinVersion;
            index_t denseIndex = 0;
            uint8_t alive = 0;
            uint8_t inactive = 0; // set when the slot version overflows and must never be reused
        };

        using SparseSlots = std::vector<SparseSlot, stl::Allocator<SparseSlot>>;
        using DenseValues = std::vector<T, stl::Allocator<T>>;
        using DenseToSparse = std::vector<index_t, stl::Allocator<index_t>>;
        using FreeIndices = std::deque<index_t, stl::Allocator<index_t>>;

        [[nodiscard]] const T* getImpl(key k) const noexcept
        {
            index_t sparseIndex = key::toIndex(k);
            if (sparseIndex >= sparseSlots.size())
            {
                return nullptr;
            }

            const SparseSlot& slot = sparseSlots[sparseIndex];
            if (slot.alive == 0 || slot.inactive != 0)
            {
                return nullptr;
            }

            version_t version = key::toVersion(k);
            if (slot.version != version || version == key::kInvalidVersion)
            {
                return nullptr;
            }

            SLOT_MAP_ASSERT(slot.denseIndex < denseValues.size());
            return &denseValues[slot.denseIndex];
        }

        [[nodiscard]] bool isValidSparseIndex(index_t sparseIndex) const noexcept
        {
            return sparseIndex < sparseSlots.size();
        }

        enum class EraseResult
        {
            NotFound,
            ErasedAndIndexRecycled,
            ErasedAndIndexDeactivated,
        };

        template <bool VERSION_CHECK>
        EraseResult eraseImpl(key k)
        {
            index_t sparseIndex = key::toIndex(k);
            if (!isValidSparseIndex(sparseIndex))
            {
                return EraseResult::NotFound;
            }

            SparseSlot& slot = sparseSlots[sparseIndex];
            if (slot.alive == 0 || slot.inactive != 0)
            {
                return EraseResult::NotFound;
            }

            version_t slotVersion = slot.version;
            if constexpr (VERSION_CHECK)
            {
                version_t keyVersion = key::toVersion(k);
                if (slotVersion != keyVersion || slotVersion == key::kInvalidVersion || keyVersion == key::kInvalidVersion)
                {
                    return EraseResult::NotFound;
                }
            }

            static_assert(std::is_move_assignable<T>::value || std::is_copy_assignable<T>::value,
                "dense_slotmap::slot_map<T>::erase requires T to be move-assignable or copy-assignable");

            index_t denseIndex = slot.denseIndex;
            index_t lastDenseIndex = static_cast<index_t>(denseValues.size() - 1u);
            SLOT_MAP_ASSERT(denseIndex <= lastDenseIndex);
            SLOT_MAP_ASSERT(denseToSparse.size() == denseValues.size());
            SLOT_MAP_ASSERT(denseToSparse[denseIndex] == sparseIndex);

            if (denseIndex != lastDenseIndex)
            {
                index_t movedSparseIndex = denseToSparse[lastDenseIndex];

                denseValues[denseIndex] = std::move(denseValues[lastDenseIndex]);
                denseToSparse[denseIndex] = movedSparseIndex;

                SparseSlot& movedSlot = sparseSlots[movedSparseIndex];
                SLOT_MAP_ASSERT(movedSlot.alive != 0);
                movedSlot.denseIndex = denseIndex;
            }

            denseValues.pop_back();
            denseToSparse.pop_back();

            bool deactivateSlot = (slotVersion == key::kMaxVersion);
            if (deactivateSlot)
            {
                slot.inactive = 1;
            }
            else
            {
                slotVersion = key::increaseVersion(slotVersion);
                SLOT_MAP_ASSERT(slotVersion != key::kInvalidVersion);
                SLOT_MAP_ASSERT(slotVersion > slot.version);
                freeIndices.emplace_back(sparseIndex);
            }

            slot.version = slotVersion;
            slot.alive = 0;
            slot.denseIndex = 0;

            return deactivateSlot ? EraseResult::ErasedAndIndexDeactivated : EraseResult::ErasedAndIndexRecycled;
        }

    public:
        slot_map() = default;
        ~slot_map() = default;

        slot_map(const slot_map&) = default;
        slot_map& operator=(const slot_map&) = default;

        slot_map(slot_map&&) noexcept = default;
        slot_map& operator=(slot_map&&) noexcept = default;

        /*
          Returns true if the slot map contains a specific key.
        */
        [[nodiscard]] bool has_key(key k) const noexcept
        {
            index_t sparseIndex = key::toIndex(k);
            if (sparseIndex >= sparseSlots.size())
            {
                return false;
            }

            const SparseSlot& slot = sparseSlots[sparseIndex];
            if (slot.alive == 0 || slot.inactive != 0)
            {
                return false;
            }

            version_t version = key::toVersion(k);
            return version != key::kInvalidVersion && slot.version == version;
        }

        [[nodiscard]] bool contains(key k) const noexcept { return has_key(k); }

        /*
          Clears the slot map and releases allocated memory.
          As with the sparse paged version, old handles may collide with future handles after reset().
        */
        void reset()
        {
            DenseValues tmpDenseValues;
            DenseToSparse tmpDenseToSparse;
            SparseSlots tmpSparseSlots;
            FreeIndices tmpFreeIndices;

            denseValues.swap(tmpDenseValues);
            denseToSparse.swap(tmpDenseToSparse);
            sparseSlots.swap(tmpSparseSlots);
            freeIndices.swap(tmpFreeIndices);
        }

        /*
          Clears live values but keeps allocated memory for reuse.
          Existing live keys are invalidated by increasing their slot versions.
        */
        void clear()
        {
            for (index_t sparseIndex : denseToSparse)
            {
                SLOT_MAP_ASSERT(sparseIndex < sparseSlots.size());
                SparseSlot& slot = sparseSlots[sparseIndex];
                SLOT_MAP_ASSERT(slot.alive != 0);

                if (slot.version == key::kMaxVersion)
                {
                    slot.inactive = 1;
                }
                else
                {
                    slot.version = key::increaseVersion(slot.version);
                    SLOT_MAP_ASSERT(slot.version != key::kInvalidVersion);
                    freeIndices.emplace_back(sparseIndex);
                }

                slot.alive = 0;
                slot.denseIndex = 0;
            }

            denseValues.clear();
            denseToSparse.clear();
        }

        /*
          Reserve dense value capacity and, optionally, sparse handle-slot capacity.
          This can reduce pointer/reference invalidation caused by emplace() reallocations,
          but erase() can still move one live value because this is a dense container.
        */
        void reserve(size_type denseCapacity, size_type sparseCapacity = 0)
        {
            denseValues.reserve(denseCapacity);
            denseToSparse.reserve(denseCapacity);
            if (sparseCapacity > 0)
            {
                sparseSlots.reserve(sparseCapacity);
            }
        }

        [[nodiscard]] const T* get(key k) const noexcept { return getImpl(k); }

        [[nodiscard]] T* get(key k) noexcept
        {
            const T* constResult = getImpl(k);
            return const_cast<T*>(constResult);
        }

        template <class... Args>
        [[nodiscard]] key emplace(Args&&... args)
        {
            bool useRecycledIndex = static_cast<size_type>(freeIndices.size()) > kMinFreeIndices;
            index_t sparseIndex = 0;

            if (useRecycledIndex)
            {
                sparseIndex = freeIndices.front();
                SLOT_MAP_ASSERT(sparseIndex < sparseSlots.size());
                SLOT_MAP_ASSERT(sparseSlots[sparseIndex].alive == 0);
                SLOT_MAP_ASSERT(sparseSlots[sparseIndex].inactive == 0);
            }
            else
            {
                SLOT_MAP_ASSERT(sparseSlots.size() <= static_cast<size_t>(key::kMaxIndex));
                sparseIndex = static_cast<index_t>(sparseSlots.size());
                sparseSlots.emplace_back();
            }

            index_t denseIndex = static_cast<index_t>(denseValues.size());
            SLOT_MAP_ASSERT(denseValues.size() <= static_cast<size_t>(key::kMaxIndex));

            try
            {
                denseValues.emplace_back(std::forward<Args>(args)...);
                try
                {
                    denseToSparse.emplace_back(sparseIndex);
                }
                catch (...)
                {
                    denseValues.pop_back();
                    throw;
                }
            }
            catch (...)
            {
                if (!useRecycledIndex)
                {
                    sparseSlots.pop_back();
                }
                throw;
            }

            if (useRecycledIndex)
            {
                freeIndices.pop_front();
            }

            SparseSlot& slot = sparseSlots[sparseIndex];
            slot.denseIndex = denseIndex;
            slot.alive = 1;
            slot.inactive = 0;

            return key::make(slot.version, sparseIndex);
        }

        void erase(key k) { eraseImpl<true>(k); }

        [[nodiscard]] std::optional<T> pop(key k)
        {
            T* value = get(k);
            if (value == nullptr)
            {
                return {};
            }

            T result(std::move(*value));
            eraseImpl<true>(k);
            return result;
        }

        [[nodiscard]] bool empty() const noexcept { return denseValues.empty(); }
        [[nodiscard]] size_type size() const noexcept { return static_cast<size_type>(denseValues.size()); }
        [[nodiscard]] size_type dense_capacity() const noexcept { return static_cast<size_type>(denseValues.capacity()); }
        [[nodiscard]] size_type sparse_capacity() const noexcept { return static_cast<size_type>(sparseSlots.capacity()); }
        [[nodiscard]] size_type sparse_size() const noexcept { return static_cast<size_type>(sparseSlots.size()); }

        [[nodiscard]] T* data() noexcept { return denseValues.data(); }
        [[nodiscard]] const T* data() const noexcept { return denseValues.data(); }

        void swap(slot_map& other) noexcept
        {
            denseValues.swap(other.denseValues);
            denseToSparse.swap(other.denseToSparse);
            sparseSlots.swap(other.sparseSlots);
            freeIndices.swap(other.freeIndices);
        }

        struct Stats
        {
            size_type numSparseSlotsTotal = 0;
            size_type numAliveItems = 0;
            size_type numTombstoneItems = 0;
            size_type numInactiveItems = 0;
            size_type numFreeIndices = 0;
            size_type numDenseCapacity = 0;
            size_type numSparseCapacity = 0;
        };

        [[nodiscard]] Stats debug_stats() const noexcept
        {
            Stats stats;
            stats.numSparseSlotsTotal = static_cast<size_type>(sparseSlots.size());
            stats.numAliveItems = static_cast<size_type>(denseValues.size());
            stats.numFreeIndices = static_cast<size_type>(freeIndices.size());
            stats.numDenseCapacity = static_cast<size_type>(denseValues.capacity());
            stats.numSparseCapacity = static_cast<size_type>(sparseSlots.capacity());

            for (const SparseSlot& slot : sparseSlots)
            {
                if (slot.inactive != 0)
                {
                    stats.numInactiveItems++;
                }
                else if (slot.alive == 0)
                {
                    stats.numTombstoneItems++;
                }
            }

            return stats;
        }

    public:
        // values iteration: for (auto& value : slotMap) { ... }
        using values_iterator = typename DenseValues::iterator;
        using const_values_iterator = typename DenseValues::const_iterator;

        [[nodiscard]] values_iterator begin() noexcept { return denseValues.begin(); }
        [[nodiscard]] values_iterator end() noexcept { return denseValues.end(); }
        [[nodiscard]] const_values_iterator begin() const noexcept { return denseValues.begin(); }
        [[nodiscard]] const_values_iterator end() const noexcept { return denseValues.end(); }
        [[nodiscard]] const_values_iterator cbegin() const noexcept { return denseValues.cbegin(); }
        [[nodiscard]] const_values_iterator cend() const noexcept { return denseValues.cend(); }

        // key-value iteration: for (const auto& kv : slotMap.items()) { ... }
        template <bool IsConst>
        class kv_iterator_impl
        {
        public:
            template <typename TYPE>
            struct reference
            {
                TYPE* ptr = nullptr;

                explicit reference(TYPE* _ptr)
                    : ptr(_ptr)
                {
                }

                void set(TYPE* _ptr) noexcept { ptr = _ptr; }

                [[nodiscard]] TYPE& get() const noexcept
                {
                    SLOT_MAP_ASSERT(ptr);
                    return *ptr;
                }

                operator TYPE& () const noexcept { return get(); }
            };

            using slot_map_ptr = std::conditional_t<IsConst, const slot_map*, slot_map*>;
            using value_ptr = std::conditional_t<IsConst, const T*, T*>;
            using KeyValue = std::conditional_t<IsConst,
                std::pair<key, const reference<const T>>,
                std::pair<key, reference<T>>>;

        private:
            void updateTmpKV() const noexcept
            {
                SLOT_MAP_ASSERT(denseIndex <= slotMap->denseValues.size());
                SLOT_MAP_ASSERT(denseIndex < slotMap->denseValues.size());

                index_t sparseIndex = slotMap->denseToSparse[denseIndex];
                const SparseSlot& slot = slotMap->sparseSlots[sparseIndex];
                SLOT_MAP_ASSERT(slot.alive != 0);
                SLOT_MAP_ASSERT(slot.denseIndex == denseIndex);

                tmpKv.first = key::make(slot.version, sparseIndex);
                value_ptr value = &slotMap->denseValues[denseIndex];

                if constexpr (IsConst)
                {
                    const_cast<reference<const T>&>(tmpKv.second).set(value);
                }
                else
                {
                    tmpKv.second.set(value);
                }
            }

        public:
            explicit kv_iterator_impl(slot_map_ptr _slotMap, size_type _denseIndex) noexcept
                : slotMap(_slotMap)
                , denseIndex(_denseIndex)
                , tmpKv(key::invalid(), std::conditional_t<IsConst, reference<const T>, reference<T>>(nullptr))
            {
            }

            template <bool OtherIsConst, typename = std::enable_if_t<IsConst && !OtherIsConst>>
            kv_iterator_impl(const kv_iterator_impl<OtherIsConst>& other) noexcept
                : slotMap(other.slotMap)
                , denseIndex(other.denseIndex)
                , tmpKv(key::invalid(), reference<const T>(nullptr))
            {
            }

            [[nodiscard]] const KeyValue& operator*() const noexcept
            {
                updateTmpKV();
                return tmpKv;
            }

            [[nodiscard]] const KeyValue* operator->() const noexcept
            {
                updateTmpKV();
                return &tmpKv;
            }

            [[nodiscard]] bool operator==(const kv_iterator_impl& other) const noexcept
            {
                return slotMap == other.slotMap && denseIndex == other.denseIndex;
            }

            [[nodiscard]] bool operator!=(const kv_iterator_impl& other) const noexcept
            {
                return !(*this == other);
            }

            kv_iterator_impl& operator++() noexcept
            {
                denseIndex++;
                return *this;
            }

            kv_iterator_impl operator++(int) noexcept
            {
                kv_iterator_impl result = *this;
                ++*this;
                return result;
            }

        private:
            slot_map_ptr slotMap = nullptr;
            size_type denseIndex = 0;
            mutable KeyValue tmpKv;

            template <bool>
            friend class kv_iterator_impl;
        };

        using kv_iterator = kv_iterator_impl<false>;
        using const_kv_iterator = kv_iterator_impl<true>;

        template <bool IsConst>
        class items_impl
        {
            using slot_map_ptr = std::conditional_t<IsConst, const slot_map*, slot_map*>;
            using iterator_type = std::conditional_t<IsConst, const_kv_iterator, kv_iterator>;

        public:
            explicit items_impl(slot_map_ptr _slotMap) noexcept
                : slotMap(_slotMap)
            {
            }

            [[nodiscard]] iterator_type begin() const noexcept { return iterator_type(slotMap, 0); }
            [[nodiscard]] iterator_type end() const noexcept { return iterator_type(slotMap, slotMap->size()); }

        private:
            slot_map_ptr slotMap = nullptr;
        };

        using Items = items_impl<true>;
        using MutableItems = items_impl<false>;

        [[nodiscard]] Items items() const noexcept { return Items(this); }
        [[nodiscard]] MutableItems items() noexcept { return MutableItems(this); }

    private:
        DenseValues denseValues;
        DenseToSparse denseToSparse;
        SparseSlots sparseSlots;
        FreeIndices freeIndices;
    };

    template <class T, size_t MINFREEINDICES = 64>
    using slot_map32 = slot_map<T, dense_slotmap::slot_map_key32<T>, MINFREEINDICES>;

    template <class T, size_t MINFREEINDICES = 64>
    using slot_map64 = slot_map<T, dense_slotmap::slot_map_key64<T>, MINFREEINDICES>;

} // namespace dense_slotmap

namespace std
{
    template <typename T>
    struct hash<dense_slotmap::slot_map_key64<T>>
    {
        size_t operator()(const dense_slotmap::slot_map_key64<T>& key) const noexcept { return key.hash(); }
    };

    template <typename T>
    struct hash<dense_slotmap::slot_map_key32<T>>
    {
        size_t operator()(const dense_slotmap::slot_map_key32<T>& key) const noexcept { return key.hash(); }
    };
} // namespace std


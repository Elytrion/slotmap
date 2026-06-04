#pragma once
// modified from original implementation from https://github.com/SergeyMakeev/SlotMap/blob/main/slot_map/slot_map.h
#include <assert.h>
#include <stdint.h>
#include <cstddef>
#include <deque>
#include <functional>
#include <vector>
#include <algorithm>
#include <optional>

// You could override memory allocator by defining SLOT_MAP_ALLOC/SLOT_MAP_FREE macroses
#if !defined(SLOT_MAP_ALLOC) || !defined(SLOT_MAP_FREE)

#if defined(_WIN32)
// Windows
#include <xmmintrin.h>
#define SLOT_MAP_ALLOC(sizeInBytes, alignment) _mm_malloc(sizeInBytes, alignment)
#define SLOT_MAP_FREE(ptr) _mm_free(ptr)
#else
// Posix
#include <stdlib.h>
#define SLOT_MAP_ALLOC(sizeInBytes, alignment) aligned_alloc(alignment, sizeInBytes)
#define SLOT_MAP_FREE(ptr) free(ptr)
#endif

#endif

// extern void _onAssertionFailed(const char* expression, const char* srcFile, unsigned int srcLine);
// #define SLOT_MAP_ASSERT(expression) (void)((!!(expression)) || (_onAssertionFailed(#expression, __FILE__, (unsigned int)(__LINE__)), 0))

// You could override asserts by defining SLOT_MAP_ASSERT macro
#if !defined(SLOT_MAP_ASSERT)
#include <assert.h>
#define SLOT_MAP_ASSERT(expression) assert(expression)
#endif

namespace stl
{
    // STL compatible allocator
    // Note: some platforms (macOS) does not support alignments smaller than `alignof(void*)`
    template <class T, size_t Alignment = std::max(alignof(T), alignof(void*))> struct Allocator
    {
    public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;
        using size_type = size_t;
        using difference_type = ptrdiff_t;

        template <class U> struct rebind
        {
            using other = Allocator<U, Alignment>;
        };

        Allocator() noexcept {}
        Allocator(const Allocator& /*other*/) noexcept {}

        template <typename U> Allocator(const Allocator<U, Alignment>& /* other */) noexcept {}

        ~Allocator() {}

        pointer address(reference x) const noexcept { return &x; }
        const_pointer address(const_reference x) const noexcept { return &x; }

        pointer allocate(size_type n, [[maybe_unused]] const void* hint = 0)
        {
            size_t alignment = Alignment;
            n = std::max(n, alignment);
            pointer p = reinterpret_cast<pointer>(SLOT_MAP_ALLOC((sizeof(value_type) * n), alignment));
            SLOT_MAP_ASSERT(p);
            return p;
        }

        void deallocate(pointer p, size_type /* n */) { SLOT_MAP_FREE(p); }

        size_type max_size() const noexcept { return std::numeric_limits<size_type>::max() / sizeof(value_type); }

        template <class U, class... Args> void construct(U* p, Args&&... args)
        {
            new (reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);
        }
        template <class U> void destroy(U* p) { p->~U(); }
    };

    template <class T1, class T2, size_t Alignment>
    bool operator==(const Allocator<T1, Alignment>& /*lhs*/, const Allocator<T2, Alignment>& /*rhs*/) noexcept
    {
        return true;
    }

    template <class T1, class T2, size_t Alignment>
    bool operator!=(const Allocator<T1, Alignment>& /*lhs*/, const Allocator<T2, Alignment>& /*rhs*/) noexcept
    {
        return false;
    }
} // namespace stl

namespace slotmap_commons
{
    /*
    Even though slot map keys are technically typeless (uint64_t), we artificially add a new type to get extra compiler checks.

    i.e., the following code should not compile
    ```
    slot_map<std::string> strings;
    slot_map<int> numbers;
    slot_map<int>::key numKey =  numbers.emplace(3);
    const std::string* value = strings.get(numKey);
    ```
    */
    /*

    64-bit key

    | Component      |  Number of bits        |
    | ---------------|------------------------|
    | version        |  32 (0..4,294,967,295) |
    | index          |  32 (0..4,294,967,295) |

    */
    template <typename T, typename Domain>
    struct slot_map_key64
    {
        using id_type = uint64_t;
        using version_t = uint32_t;
        using index_t = uint32_t;

        static inline constexpr version_t kInvalidVersion = 0x0u;
        static inline constexpr version_t kMinVersion = 0x1u;
        static inline constexpr version_t kMaxVersion = 0xffffffffu;
        static inline constexpr index_t   kMaxIndex = 0xffffffffu;

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

        [[nodiscard]] inline size_t hash() const noexcept
        {
            return std::hash<id_type>{}(raw);
        }

        [[nodiscard]] static inline slot_map_key64 updateVersion(slot_map_key64 key, version_t version) noexcept
        {
            SLOT_MAP_ASSERT(version != kInvalidVersion);

            id_type ver = (static_cast<id_type>(version) << kVersionShift) & kVersionMask;
            return slot_map_key64{ (key.raw & ~kVersionMask) | ver };
        }

        [[nodiscard]] static inline index_t toIndex(slot_map_key64 key) noexcept
        {
            return static_cast<index_t>(key.raw & kIndexMask);
        }

        [[nodiscard]] static inline version_t toVersion(slot_map_key64 key) noexcept
        {
            return static_cast<version_t>((key.raw & kVersionMask) >> kVersionShift);
        }

        [[nodiscard]] static inline version_t increaseVersion(version_t version) noexcept
        {
            return version + 1;
        }

        slot_map_key64() noexcept = default;

        explicit slot_map_key64(id_type raw) noexcept
            : raw(raw)
        {
        }

        bool operator==(const slot_map_key64& other) const noexcept
        {
            return raw == other.raw;
        }

        bool operator<(const slot_map_key64& other) const noexcept
        {
            return raw < other.raw;
        }

        [[nodiscard]] explicit operator id_type() const noexcept
        {
            return raw;
        }

        static inline slot_map_key64 invalid() noexcept
        {
            return slot_map_key64{ 0 };
        }

        id_type raw = 0;
    };

    /*

    32-bit key

    | Component      |  Number of bits     |
    | ---------------|---------------------|
    | version        |  12 (0..4095)       |
    | index          |  20 (0..1,048,576)  |

    */
    template <typename T, typename Domain>
    struct slot_map_key32
    {
        using id_type = uint32_t;
        using version_t = uint16_t;
        using index_t = uint32_t;

        static inline constexpr version_t kInvalidVersion = 0x0u;
        static inline constexpr version_t kMinVersion = 0x1u;
        static inline constexpr version_t kMaxVersion = 0x0fffu;
        static inline constexpr index_t   kMaxIndex = 0x000fffffu;

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

        [[nodiscard]] inline size_t hash() const noexcept
        {
            return std::hash<id_type>{}(raw);
        }

        [[nodiscard]] static inline slot_map_key32 updateVersion(slot_map_key32 key, version_t version) noexcept
        {
            SLOT_MAP_ASSERT(version != kInvalidVersion);
            SLOT_MAP_ASSERT(version <= kMaxVersion);

            id_type ver = (static_cast<id_type>(version) << kVersionShift) & kVersionMask;
            return slot_map_key32{ (key.raw & ~kVersionMask) | ver };
        }

        [[nodiscard]] static inline index_t toIndex(slot_map_key32 key) noexcept
        {
            return static_cast<index_t>(key.raw & kIndexMask);
        }

        [[nodiscard]] static inline version_t toVersion(slot_map_key32 key) noexcept
        {
            return static_cast<version_t>((key.raw & kVersionMask) >> kVersionShift);
        }

        [[nodiscard]] static inline version_t increaseVersion(version_t version) noexcept
        {
            return version + 1;
        }

        slot_map_key32() noexcept = default;

        explicit slot_map_key32(id_type raw) noexcept
            : raw(raw)
        {
        }

        bool operator==(const slot_map_key32& other) const noexcept
        {
            return raw == other.raw;
        }

        bool operator<(const slot_map_key32& other) const noexcept
        {
            return raw < other.raw;
        }

        [[nodiscard]] explicit operator id_type() const noexcept
        {
            return raw;
        }

        static inline slot_map_key32 invalid() noexcept
        {
            return slot_map_key32{ 0 };
        }

        id_type raw = 0;
    };
}

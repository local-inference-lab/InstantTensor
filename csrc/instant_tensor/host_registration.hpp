#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace instanttensor {

struct HostRegistrationRange {
    void* ptr;
    size_t size;
};

struct HostRegistration {
    std::vector<HostRegistrationRange> ranges;
    int whole_buffer_error = 0;
};

class HostRegistrationCleanupError : public std::runtime_error {
public:
    // Storage may still be registered with CUDA. Transfer these ranges to a
    // recovery handler; never free the allocation merely by destroying this error.
    // Unrecovered storage remains allocated until process exit.
    void* allocation_ptr;
    HostRegistration registration;
    int cleanup_error;

    HostRegistrationCleanupError(void* ptr, HostRegistration remaining, int error)
        : std::runtime_error("Host registration rollback failed; allocation retained, pinned-memory fallback disabled"),
          allocation_ptr(ptr), registration(std::move(remaining)), cleanup_error(error) {}
};

struct HostBufferAllocation {
    void* ptr = nullptr;
    HostRegistration registration;
    bool runtime_allocated = false;
    std::string registration_failure;
};

template <typename RegisterFn, typename UnregisterFn, typename ClearErrorFn>
HostRegistration register_host_buffer(
    void* ptr,
    size_t size,
    unsigned int flags,
    size_t segment_size,
    RegisterFn&& register_fn,
    UnregisterFn&& unregister_fn,
    ClearErrorFn&& clear_error_fn
) {
    if (ptr == nullptr || size == 0 || segment_size == 0) {
        throw std::invalid_argument("Host registration requires non-empty storage and segments");
    }

    HostRegistration registration;
    // Reserve bookkeeping before registering storage so allocation failure
    // cannot discard ownership of an already registered segment.
    registration.ranges.reserve(1 + (size - 1) / segment_size);
    int result = register_fn(ptr, size, flags);
    if (result == 0) {
        registration.ranges.push_back({ptr, size});
        return registration;
    }

    registration.whole_buffer_error = result;
    clear_error_fn();
    auto* base = static_cast<char*>(ptr);
    for (size_t offset = 0; offset < size; offset += segment_size) {
        const size_t current_size = std::min(segment_size, size - offset);
        void* current_ptr = base + offset;
        result = register_fn(current_ptr, current_size, flags);
        if (result == 0) {
            registration.ranges.push_back({current_ptr, current_size});
            continue;
        }

        clear_error_fn();
        int cleanup_error = 0;
        for (size_t index = registration.ranges.size(); index > 0; --index) {
            const int error = unregister_fn(registration.ranges[index - 1].ptr);
            if (error == 0) {
                registration.ranges.erase(registration.ranges.begin() + index - 1);
            } else {
                cleanup_error = error;
            }
        }
        if (cleanup_error != 0) {
            throw HostRegistrationCleanupError(ptr, std::move(registration), cleanup_error);
        }
        throw std::runtime_error(
            "Host registration failed for the whole buffer (code "
            + std::to_string(registration.whole_buffer_error)
            + ") and segment at offset " + std::to_string(offset)
            + " (code " + std::to_string(result) + ")"
        );
    }
    return registration;
}

template <
    typename AlignedAllocFn,
    typename FreeFn,
    typename RegisterFn,
    typename UnregisterFn,
    typename ClearErrorFn,
    typename RuntimeAllocFn>
HostBufferAllocation allocate_registered_host_buffer(
    size_t size,
    size_t alignment,
    unsigned int register_flags,
    unsigned int runtime_alloc_flags,
    size_t segment_size,
    AlignedAllocFn&& aligned_alloc_fn,
    FreeFn&& free_fn,
    RegisterFn&& register_fn,
    UnregisterFn&& unregister_fn,
    ClearErrorFn&& clear_error_fn,
    RuntimeAllocFn&& runtime_alloc_fn
) {
    if (size == 0 || alignment == 0) {
        throw std::invalid_argument("Host allocation requires non-empty aligned storage");
    }

    HostBufferAllocation allocation;
    allocation.ptr = aligned_alloc_fn(alignment, size);
    if (allocation.ptr == nullptr) {
        throw std::runtime_error("Failed to allocate aligned host storage");
    }

    try {
        allocation.registration = register_host_buffer(
            allocation.ptr,
            size,
            register_flags,
            segment_size,
            std::forward<RegisterFn>(register_fn),
            std::forward<UnregisterFn>(unregister_fn),
            std::forward<ClearErrorFn>(clear_error_fn)
        );
        return allocation;
    } catch (const HostRegistrationCleanupError&) {
        throw;
    } catch (const std::exception& error) {
        allocation.registration_failure = error.what();
    }

    free_fn(allocation.ptr);
    allocation.ptr = nullptr;
    clear_error_fn();

    const int result = runtime_alloc_fn(
        &allocation.ptr,
        size,
        runtime_alloc_flags
    );
    if (result != 0 || allocation.ptr == nullptr) {
        throw std::runtime_error(
            allocation.registration_failure
            + "; runtime pinned allocation failed (code "
            + std::to_string(result) + ")"
        );
    }

    allocation.runtime_allocated = true;
    return allocation;
}

} // namespace instanttensor

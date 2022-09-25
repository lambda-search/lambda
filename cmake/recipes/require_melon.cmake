
find_path(MELON_INCLUDE_PATH NAMES "melon/base/profile.h")
find_library(MELON_LIBRARY NAMES melon)
include_directories(${MELON_INCLUDE_PATH})
if((NOT MELON_INCLUDE_PATH) OR (NOT MELON_LIBRARY))
    message(FATAL_ERROR "Fail to find melon")
endif()

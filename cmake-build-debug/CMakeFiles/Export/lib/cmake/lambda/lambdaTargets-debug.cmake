#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "lambda::ann" for configuration "Debug"
set_property(TARGET lambda::ann APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(lambda::ann PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libann.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libann.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS lambda::ann )
list(APPEND _IMPORT_CHECK_FILES_FOR_lambda::ann "${_IMPORT_PREFIX}/lib/libann.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

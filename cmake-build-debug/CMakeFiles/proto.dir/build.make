# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/liyinbin/github/lambda-search/lambda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/liyinbin/github/lambda-search/lambda/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/proto.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/proto.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/proto.dir/flags.make

lambda/proto/types.pb.h: ../lambda/proto/types.proto
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/liyinbin/github/lambda-search/lambda/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating lambda/proto/types.pb.h, lambda/proto/types.pb.cc"
	/Users/liyinbin/miniconda3/envs/karabor-dev/bin/protoc -I/Users/liyinbin/miniconda3/envs/karabor-dev/include -I/Users/liyinbin/github/lambda-search/lambda -I/Users/liyinbin/github/lambda-search/lambda --cpp_out=/Users/liyinbin/github/lambda-search/lambda/cmake-build-debug /Users/liyinbin/github/lambda-search/lambda/lambda/proto/types.proto

lambda/proto/types.pb.cc: lambda/proto/types.pb.h
	@$(CMAKE_COMMAND) -E touch_nocreate lambda/proto/types.pb.cc

CMakeFiles/proto.dir/lambda/proto/types.pb.cc.o: CMakeFiles/proto.dir/flags.make
CMakeFiles/proto.dir/lambda/proto/types.pb.cc.o: lambda/proto/types.pb.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/liyinbin/github/lambda-search/lambda/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/proto.dir/lambda/proto/types.pb.cc.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/proto.dir/lambda/proto/types.pb.cc.o -c /Users/liyinbin/github/lambda-search/lambda/cmake-build-debug/lambda/proto/types.pb.cc

CMakeFiles/proto.dir/lambda/proto/types.pb.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/proto.dir/lambda/proto/types.pb.cc.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/liyinbin/github/lambda-search/lambda/cmake-build-debug/lambda/proto/types.pb.cc > CMakeFiles/proto.dir/lambda/proto/types.pb.cc.i

CMakeFiles/proto.dir/lambda/proto/types.pb.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/proto.dir/lambda/proto/types.pb.cc.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/liyinbin/github/lambda-search/lambda/cmake-build-debug/lambda/proto/types.pb.cc -o CMakeFiles/proto.dir/lambda/proto/types.pb.cc.s

# Object files for target proto
proto_OBJECTS = \
"CMakeFiles/proto.dir/lambda/proto/types.pb.cc.o"

# External object files for target proto
proto_EXTERNAL_OBJECTS =

lib/libproto.a: CMakeFiles/proto.dir/lambda/proto/types.pb.cc.o
lib/libproto.a: CMakeFiles/proto.dir/build.make
lib/libproto.a: CMakeFiles/proto.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/liyinbin/github/lambda-search/lambda/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library lib/libproto.a"
	$(CMAKE_COMMAND) -P CMakeFiles/proto.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/proto.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/proto.dir/build: lib/libproto.a

.PHONY : CMakeFiles/proto.dir/build

CMakeFiles/proto.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/proto.dir/cmake_clean.cmake
.PHONY : CMakeFiles/proto.dir/clean

CMakeFiles/proto.dir/depend: lambda/proto/types.pb.cc
CMakeFiles/proto.dir/depend: lambda/proto/types.pb.h
	cd /Users/liyinbin/github/lambda-search/lambda/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/liyinbin/github/lambda-search/lambda /Users/liyinbin/github/lambda-search/lambda /Users/liyinbin/github/lambda-search/lambda/cmake-build-debug /Users/liyinbin/github/lambda-search/lambda/cmake-build-debug /Users/liyinbin/github/lambda-search/lambda/cmake-build-debug/CMakeFiles/proto.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/proto.dir/depend


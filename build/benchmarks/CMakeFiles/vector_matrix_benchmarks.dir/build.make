# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/simi/Dev/linear-algebra-lib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/simi/Dev/linear-algebra-lib/build

# Include any dependencies generated for this target.
include benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/compiler_depend.make

# Include the progress variables for this target.
include benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/progress.make

# Include the compile flags for this target's objects.
include benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/flags.make

benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/codegen:
.PHONY : benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/codegen

benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/vector_matrix_benchmark.cpp.o: benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/flags.make
benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/vector_matrix_benchmark.cpp.o: /home/simi/Dev/linear-algebra-lib/benchmarks/vector_matrix_benchmark.cpp
benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/vector_matrix_benchmark.cpp.o: benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/simi/Dev/linear-algebra-lib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/vector_matrix_benchmark.cpp.o"
	cd /home/simi/Dev/linear-algebra-lib/build/benchmarks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/vector_matrix_benchmark.cpp.o -MF CMakeFiles/vector_matrix_benchmarks.dir/vector_matrix_benchmark.cpp.o.d -o CMakeFiles/vector_matrix_benchmarks.dir/vector_matrix_benchmark.cpp.o -c /home/simi/Dev/linear-algebra-lib/benchmarks/vector_matrix_benchmark.cpp

benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/vector_matrix_benchmark.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/vector_matrix_benchmarks.dir/vector_matrix_benchmark.cpp.i"
	cd /home/simi/Dev/linear-algebra-lib/build/benchmarks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/simi/Dev/linear-algebra-lib/benchmarks/vector_matrix_benchmark.cpp > CMakeFiles/vector_matrix_benchmarks.dir/vector_matrix_benchmark.cpp.i

benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/vector_matrix_benchmark.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/vector_matrix_benchmarks.dir/vector_matrix_benchmark.cpp.s"
	cd /home/simi/Dev/linear-algebra-lib/build/benchmarks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/simi/Dev/linear-algebra-lib/benchmarks/vector_matrix_benchmark.cpp -o CMakeFiles/vector_matrix_benchmarks.dir/vector_matrix_benchmark.cpp.s

# Object files for target vector_matrix_benchmarks
vector_matrix_benchmarks_OBJECTS = \
"CMakeFiles/vector_matrix_benchmarks.dir/vector_matrix_benchmark.cpp.o"

# External object files for target vector_matrix_benchmarks
vector_matrix_benchmarks_EXTERNAL_OBJECTS =

benchmarks/vector_matrix_benchmarks: benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/vector_matrix_benchmark.cpp.o
benchmarks/vector_matrix_benchmarks: benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/build.make
benchmarks/vector_matrix_benchmarks: benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/compiler_depend.ts
benchmarks/vector_matrix_benchmarks: /usr/lib/libbenchmark.so.1.9.1
benchmarks/vector_matrix_benchmarks: benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/simi/Dev/linear-algebra-lib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable vector_matrix_benchmarks"
	cd /home/simi/Dev/linear-algebra-lib/build/benchmarks && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vector_matrix_benchmarks.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/build: benchmarks/vector_matrix_benchmarks
.PHONY : benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/build

benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/clean:
	cd /home/simi/Dev/linear-algebra-lib/build/benchmarks && $(CMAKE_COMMAND) -P CMakeFiles/vector_matrix_benchmarks.dir/cmake_clean.cmake
.PHONY : benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/clean

benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/depend:
	cd /home/simi/Dev/linear-algebra-lib/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/simi/Dev/linear-algebra-lib /home/simi/Dev/linear-algebra-lib/benchmarks /home/simi/Dev/linear-algebra-lib/build /home/simi/Dev/linear-algebra-lib/build/benchmarks /home/simi/Dev/linear-algebra-lib/build/benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : benchmarks/CMakeFiles/vector_matrix_benchmarks.dir/depend


# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vergil/Desktop/SPC

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vergil/Desktop/SPC/build

# Include any dependencies generated for this target.
include CMakeFiles/SPC.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/SPC.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/SPC.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SPC.dir/flags.make

CMakeFiles/SPC.dir/SPC.cpp.o: CMakeFiles/SPC.dir/flags.make
CMakeFiles/SPC.dir/SPC.cpp.o: /home/vergil/Desktop/SPC/SPC.cpp
CMakeFiles/SPC.dir/SPC.cpp.o: CMakeFiles/SPC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vergil/Desktop/SPC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SPC.dir/SPC.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SPC.dir/SPC.cpp.o -MF CMakeFiles/SPC.dir/SPC.cpp.o.d -o CMakeFiles/SPC.dir/SPC.cpp.o -c /home/vergil/Desktop/SPC/SPC.cpp

CMakeFiles/SPC.dir/SPC.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/SPC.dir/SPC.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vergil/Desktop/SPC/SPC.cpp > CMakeFiles/SPC.dir/SPC.cpp.i

CMakeFiles/SPC.dir/SPC.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/SPC.dir/SPC.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vergil/Desktop/SPC/SPC.cpp -o CMakeFiles/SPC.dir/SPC.cpp.s

# Object files for target SPC
SPC_OBJECTS = \
"CMakeFiles/SPC.dir/SPC.cpp.o"

# External object files for target SPC
SPC_EXTERNAL_OBJECTS =

SPC: CMakeFiles/SPC.dir/SPC.cpp.o
SPC: CMakeFiles/SPC.dir/build.make
SPC: /usr/lib/aarch64-linux-gnu/libboost_system.so
SPC: /usr/lib/aarch64-linux-gnu/libboost_thread.so
SPC: /usr/lib/aarch64-linux-gnu/libboost_chrono.so
SPC: /usr/lib/aarch64-linux-gnu/libboost_date_time.so
SPC: /usr/lib/aarch64-linux-gnu/libboost_atomic.so
SPC: /usr/lib/aarch64-linux-gnu/libQt5SerialPort.so.5.9.5
SPC: /usr/lib/aarch64-linux-gnu/libQt5Core.so.5.9.5
SPC: CMakeFiles/SPC.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/vergil/Desktop/SPC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable SPC"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SPC.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SPC.dir/build: SPC
.PHONY : CMakeFiles/SPC.dir/build

CMakeFiles/SPC.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SPC.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SPC.dir/clean

CMakeFiles/SPC.dir/depend:
	cd /home/vergil/Desktop/SPC/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vergil/Desktop/SPC /home/vergil/Desktop/SPC /home/vergil/Desktop/SPC/build /home/vergil/Desktop/SPC/build /home/vergil/Desktop/SPC/build/CMakeFiles/SPC.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/SPC.dir/depend


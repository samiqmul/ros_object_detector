# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mustafar/catkin_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mustafar/catkin_ws/src

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target list_install_components
list_install_components:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Available install components are: \"Unspecified\""
.PHONY : list_install_components

# Special rule for the target list_install_components
list_install_components/fast: list_install_components

.PHONY : list_install_components/fast

# Special rule for the target install
install: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/bin/cmake -P cmake_install.cmake
.PHONY : install

# Special rule for the target install
install/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/bin/cmake -P cmake_install.cmake
.PHONY : install/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# Special rule for the target install/strip
install/strip: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing the project stripped..."
	/usr/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip

# Special rule for the target install/strip
install/strip/fast: install/strip

.PHONY : install/strip/fast

# Special rule for the target install/local
install/local: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/usr/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local

# Special rule for the target install/local
install/local/fast: install/local

.PHONY : install/local/fast

# Special rule for the target test
test:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running tests..."
	/usr/bin/ctest --force-new-ctest-process $(ARGS)
.PHONY : test

# Special rule for the target test
test/fast: test

.PHONY : test/fast

# The main all target
all: cmake_check_build_system
	cd /home/mustafar/catkin_ws/src && $(CMAKE_COMMAND) -E cmake_progress_start /home/mustafar/catkin_ws/src/CMakeFiles /home/mustafar/catkin_ws/src/ros_object_detector/CMakeFiles/progress.marks
	cd /home/mustafar/catkin_ws/src && $(MAKE) -f CMakeFiles/Makefile2 ros_object_detector/all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/mustafar/catkin_ws/src/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	cd /home/mustafar/catkin_ws/src && $(MAKE) -f CMakeFiles/Makefile2 ros_object_detector/clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	cd /home/mustafar/catkin_ws/src && $(MAKE) -f CMakeFiles/Makefile2 ros_object_detector/preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	cd /home/mustafar/catkin_ws/src && $(MAKE) -f CMakeFiles/Makefile2 ros_object_detector/preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	cd /home/mustafar/catkin_ws/src && $(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

# Convenience name for target.
ros_object_detector/CMakeFiles/find_object.dir/rule:
	cd /home/mustafar/catkin_ws/src && $(MAKE) -f CMakeFiles/Makefile2 ros_object_detector/CMakeFiles/find_object.dir/rule
.PHONY : ros_object_detector/CMakeFiles/find_object.dir/rule

# Convenience name for target.
find_object: ros_object_detector/CMakeFiles/find_object.dir/rule

.PHONY : find_object

# fast build rule for target.
find_object/fast:
	cd /home/mustafar/catkin_ws/src && $(MAKE) -f ros_object_detector/CMakeFiles/find_object.dir/build.make ros_object_detector/CMakeFiles/find_object.dir/build
.PHONY : find_object/fast

src/find_object.o: src/find_object.cpp.o

.PHONY : src/find_object.o

# target to build an object file
src/find_object.cpp.o:
	cd /home/mustafar/catkin_ws/src && $(MAKE) -f ros_object_detector/CMakeFiles/find_object.dir/build.make ros_object_detector/CMakeFiles/find_object.dir/src/find_object.cpp.o
.PHONY : src/find_object.cpp.o

src/find_object.i: src/find_object.cpp.i

.PHONY : src/find_object.i

# target to preprocess a source file
src/find_object.cpp.i:
	cd /home/mustafar/catkin_ws/src && $(MAKE) -f ros_object_detector/CMakeFiles/find_object.dir/build.make ros_object_detector/CMakeFiles/find_object.dir/src/find_object.cpp.i
.PHONY : src/find_object.cpp.i

src/find_object.s: src/find_object.cpp.s

.PHONY : src/find_object.s

# target to generate assembly for a file
src/find_object.cpp.s:
	cd /home/mustafar/catkin_ws/src && $(MAKE) -f ros_object_detector/CMakeFiles/find_object.dir/build.make ros_object_detector/CMakeFiles/find_object.dir/src/find_object.cpp.s
.PHONY : src/find_object.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... list_install_components"
	@echo "... install"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... find_object"
	@echo "... install/strip"
	@echo "... install/local"
	@echo "... test"
	@echo "... src/find_object.o"
	@echo "... src/find_object.i"
	@echo "... src/find_object.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	cd /home/mustafar/catkin_ws/src && $(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system


# Overview 

This repository contains the (first) assignments for "Computer Vision" (`710.120`) and "Computergrafik und - vision" (`INH.03130UF`) and "Computer Vision 1" (`DGJ.40300UF`).

## Requirements

We build and run the code on an Ubuntu `20.04` machine with the default versions of OpenCV (`4.2.0`), g++ (`9.3.0`) and CMake (`3.16.3`) from the Ubuntu package archive.

To install the needed packages, just type:
```bash
sudo apt-get install build-essential cmake libopencv-dev
```
To test your code in this environment you simply need to push your code into the git repository (git push origin master).

Every push will be compiled and tested in our test system. We also store the output and generated images of your program.
As your code will be graded in this testing environment, we strongly recommend that you check that your program compiles 
and runs as you intended (i.e. produces the same output images).
To view the output and generated images of your program, just click on the `CI/CD tab -> Pipelines`. For every commit,
which you push into the repository, a test run is created. The images of the test runs
are stored as artifacts. Just click on the test run and then you should see the output of your program. On the right 
side of your screen there should be an artifacts link.

We also provide a [virtual box image](https://cloud.tugraz.at/index.php/s/HbsJjH4KjMyBCxS) with a pre-installed Ubuntu 20.04. 

## Compiling the Code

We use cmake to build our framework. If you are with a linux shell at the root of your repository, just type:
```bash
# from repopath
cd src/
mkdir build
cd build
cmake ../
make
```

To run your program, just type:
```bash
# from repopath/src/build
cd ../cv/task1/
../../build/cv/task1/cvtask1 tests/<testcase_name>.json
```
where `testcase_name` has to be one of: `tugraz, one_way, graz, coffee_shop`. 

Our server's CPUs used in our submission do not support AVX instructions, therefore it might be necessary to run a 
test case via
```bash
OPENCV_CPU_DISABLE=AVX2,AVX ../../build/cv/task1/cvtask1 tests/<testcase_name>.json
```
to get the exact same results (and thus, white diffs). However, please note that in any way, the results on submission
system are the ones which count.

## Generate Local Diffs

To generate local diffs between the reference and output images just run the following script: 
```bash
./test_all_x64.sh
```
If you have no permissions to run it, change the permission with the following command: 
```bash
sudo chmod +x test_all_x64.sh
```
## Git Basics / Creating the `submission` Branch

The following commands add the local change (of a file, in our case `algorithms.cpp`) in the working directory to 
so-called staging area and create a commit:

```bash
git add <path_to_algorithms.cpp>
git commit -m "I did this and that and enter info about it here here"
```

To push this commit to the server (and make the changes visible in the web interface), you need to `git push`:

```bash
git push origin <branch_name>
```

To create the `submission` branch (and thus make a submission), you can use the following commands from within the root 
directory of this repository:

```bash
git checkout -b submission
git push origin submission
```

The first command creates a new `submission` branch, and the second command pushes this branch onto the remote repository.

Now you will see the submission branch in your gitlab webinterface. **Double check whether it is your final state.**
If you are not familiar with git, you'll find a lot of helpful tutorials online. For example this one has a nice 
visualization helping to understand the core concepts: https://learngitbranching.js.org/.



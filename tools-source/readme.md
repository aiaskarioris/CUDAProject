# Test Tools
This directory contains the source code for the tools developed for debugging and testing the project. Building the **test-tools** target will compile these simple programs to the **tools** directory.

---

# Gen
The **gen** program is used to work with the input matrices used by the main program. It can generate matrices with the specified dimensions as well as read and display metadata from test files using the *-d/-l* options. Calling **gen** without any arguments will display a simple usage page.

# Check
This tool was used mostly on the earlier developmental stages of the project to verify that the main program was calculating the matrices correctly. This is done by making the same computations as with the parallel, main program but with serial code. The result of the serial calculations is compared with the result of *export_results()* in main. This tool isn't very important.
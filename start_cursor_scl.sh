#!/bin/bash

# Script to start Cursor/VS Code with SCL GCC toolset environment
# Usage: ./start_cursor_scl.sh

echo "Starting Cursor with GCC toolset-10 environment..."

# Set up SCL environment variables
export PATH="/opt/rh/gcc-toolset-10/root/usr/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rh/gcc-toolset-10/root/usr/lib64:/opt/rh/gcc-toolset-10/root/usr/lib:$LD_LIBRARY_PATH"
export CC="/opt/rh/gcc-toolset-10/root/usr/bin/gcc"
export CXX="/opt/rh/gcc-toolset-10/root/usr/bin/g++"
export MANPATH="/opt/rh/gcc-toolset-10/root/usr/share/man:"
export INFOPATH="/opt/rh/gcc-toolset-10/root/usr/share/info"
export PCP_DIR="/opt/rh/gcc-toolset-10/root"
export PERL5LIB="/opt/rh/gcc-toolset-10/root//usr/lib64/perl5/vendor_perl:/opt/rh/gcc-toolset-10/root/usr/lib/perl5:/opt/rh/gcc-toolset-10/root//usr/share/perl5/vendor_perl"
export PYTHONPATH="$(pwd)/src"

# Print environment info
echo "Environment configured:"
echo "  GCC version: $(gcc --version | head -n1)"
echo "  Python version: $(python3 --version)"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  CC: $CC"
echo "  CXX: $CXX"
echo ""

# Try different ways to launch Cursor/VS Code
if command -v cursor &> /dev/null; then
    echo "Launching Cursor..."
    cursor .
elif command -v code &> /dev/null; then
    echo "Launching VS Code..."
    code .
elif command -v code-insiders &> /dev/null; then
    echo "Launching VS Code Insiders..."
    code-insiders .
else
    echo "Error: Could not find Cursor, VS Code, or VS Code Insiders in PATH"
    echo "Available editors:"
    which code cursor code-insiders 2>/dev/null || echo "  None found"
    echo ""
    echo "You can manually start your editor with these environment variables:"
    echo "  CC=$CC"
    echo "  CXX=$CXX"  
    echo "  PYTHONPATH=$PYTHONPATH"
    echo "  PATH=$PATH"
    exit 1
fi
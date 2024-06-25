#!/bin/sh

# Check the number of arguments provided
if [ "$#" -eq 3 ]; then
  # If there are three arguments, pass all three to the Python script
  python3 sbp.py "$1" "$2" "$3"
elif [ "$#" -eq 2 ]; then
  # If there are two arguments, pass both to the Python script
  python3 sbp.py "$1" "$2"
else
  # If there is one argument, pass it to the Python script
  python3 sbp.py "$1"
fi


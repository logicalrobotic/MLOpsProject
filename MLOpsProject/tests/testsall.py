#This file is used to run all the tests in the tests folder. But all test could be run by run one of the other files in the test folder.
#So this is just for simplicity.
import subprocess

def run_pytest(test_scripts):
    #Build the command to run Pytest on the specified test scripts
    command = ['pytest'] + test_scripts

    try:
        #Run Pytest using subprocess
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Pytest execution failed with return code {e.returncode}")
        exit(1)

if __name__ == "__main__":
    #scripts to test
    test_scripts = ["MLOpsProject/tests/test_data.py", "MLOpsProject/tests/test_modelstructure.py", "MLOpsProject/tests/test_model.py"]

    #Run test
    run_pytest(test_scripts)
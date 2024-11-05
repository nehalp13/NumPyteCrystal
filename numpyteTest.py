import nbformat
import unittest
import numpy as np

# Function to extract code cells from a notebook
def extract_code_cells(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as nb_file:
        nb_contents = nbformat.read(nb_file, as_version=4)
    code_cells = [cell['source'] for cell in nb_contents.cells if cell.cell_type == 'code']
    return code_cells

# Function to dynamically execute extracted code
# All code will execute. Make sure any calls to functions is wrapped in `if __name__ == "__main__": `
def execute_code(code_cells):
    code_snippets = "\n".join(code_cells)
    exec_globals = {}
    exec(code_snippets, exec_globals)
    return exec_globals


# Provided test case
class TestNumpyte(unittest.TestCase):
    # This class method will extract solutions written between tags and stores it in a list
    # This allows you to access the solutions directly and check for exact text written
    # Solutions are stored in top down order
    @classmethod
    def setUpClass(cls):
        # Load the notebook
        with open('numpyte.ipynb', 'r', encoding='utf-8') as f:
            cls.nb = nbformat.read(f, as_version=4)
        
        # Extract solution blocks
        cls.solutions = []
        for cell in cls.nb.cells:
            if cell.cell_type == 'code' or cell.cell_type == 'markdown':
                source = cell.source
                if '### BEGIN SOLUTION' in source and '### END SOLUTION' in source:
                    solution_block = source.split('### BEGIN SOLUTION')[1].split('### END SOLUTION')[0].strip()
                    cls.solutions.append(solution_block)

        # Extract test blocks
        cls.tests = []
        for cell in cls.nb.cells:
            if cell.cell_type == 'code':
                source = cell.source
                if '### BEGIN HIDDEN TESTS' in source and '### END HIDDEN TESTS' in source:
                    solution_block = source.split('### BEGIN HIDDEN TESTS')[1].split('### END HIDDEN TESTS')[0].strip()
                    cls.tests.append(solution_block)


    def test_check_for_changes(self):
        specific_text = '''assert isinstance(sw_vial, np.ndarray), "sw_vial must be a NumPy array!"
assert sw_vial.shape == (10,), "sw_vial must be a 1D array of length 10!"
assert np.all(sw_vial == 0), "sw_vial must be an array of zeros!"'''
        self.assertIn(specific_text, self.tests[0], "Changes detected in the first test block for Task 1")

        specific_text = '''assert sw_vial.shape == (16,), "sw_vial must be a 1D array of length 16!"
assert np.all(sw_vial == 0), "sw_vial must be an array of zeros!"
assert "sw_ampoule" not in locals(), "sw_ampoule must be deleted!"'''
        self.assertIn(specific_text, self.tests[1], "Changes detected in the first test block for Task 2")
                
        specific_text = '''assert sw_vial.shape == (8, 2), "sw_vial must be a 2D array of shape (8, 2)!"
assert np.all(sw_vial == 0), "sw_vial must be an array of zeros!"'''
        self.assertIn(specific_text, self.tests[2], "Changes detected in the first test block for Task 3")

        specific_text = '''assert isinstance(bridge_indices, np.ndarray), "bridge_indices must be a NumPy array!"
assert bridge_indices.shape == (2,), "bridge_indices must be a 1D array of length 2!"'''
        self.assertIn(specific_text, self.tests[3], "Changes detected in the first test block for Task 4")

        specific_text = '''assert isinstance(translated_signal, list), "translated_signal must be a list!"
assert len(translated_signal) == len(signal), "translated_signal must be the same length as signal!"'''
        self.assertIn(specific_text, self.tests[4], "Changes detected in the first test block for Task 5")

        specific_text = '''assert isinstance(solution_flask, np.ndarray), "solution_flask must be a NumPy array!"
assert solution_flask.shape == (8, 4), "solution_flask must be a 2D array of shape (8, 4)!"
assert np.isnan(solution_flask).any() == False, "solution_flask should not contain any NaN values!"
assert np.isinf(solution_flask).any() == False, "solution_flask should not contain any infinite values!"
assert (solution_flask >= 0).all(), "solution_flask should not contain any negative values!"'''
        self.assertIn(specific_text, self.tests[5], "Changes detected in the first test block for Task 6")

        specific_text = '''assert isinstance(distilled_solution, np.float64), "distilled_solution must be a NumPy float64!"
assert int(distilled_solution) == 24, "distilled_solution must be equal to the answer!"'''
        self.assertIn(specific_text, self.tests[6], "Changes detected in the first test block for Task 7")

    
    #This is the general sample test placeholder                
    def test_task_1(self):
        global exec_globals  # Access the global variable
        exec(self.solutions[0], exec_globals)  # Execute the solution code
        self.assertIsInstance(exec_globals['sw_vial'], np.ndarray, "sw_vial must be a NumPy array!")
        self.assertEqual(exec_globals['sw_vial'].shape, (10,), "sw_vial must be a 1D array of length 10!")
        self.assertEqual(np.all(exec_globals['sw_vial']), 0, "sw_vial must be an array of zeros!")


     #This is the general sample test placeholder                
    def test_task_2(self):
        global exec_globals  # Access the global variable
        exec_globals["sw_ampoule"] = np.zeros(6)  # Create a variable to check if it is deleted
        exec(self.solutions[1], exec_globals)  # Execute the solution code
        self.assertIsInstance(exec_globals['sw_vial'], np.ndarray, "sw_vial must be a NumPy array!")
        self.assertEqual(exec_globals['sw_vial'].shape, (16,), "sw_vial must be a 1D array of length 16!")
        self.assertEqual(np.all(exec_globals['sw_vial']),0, "sw_vial must be an array of zeros!")
        self.assertNotIn("sw_ampoule", locals(), "sw_ampoule must be deleted!")
     
    def test_task_3(self):
        global exec_globals  # Access the global variable
        exec(self.solutions[2], exec_globals)  # Execute the solution code
        self.assertIsInstance(exec_globals['sw_vial'], np.ndarray, "sw_vial must be a NumPy array!")
        self.assertEqual(exec_globals['sw_vial'].shape, (8, 2), "sw_vial must be a 2D array of shape (8, 2)!")
        self.assertEqual(np.all(exec_globals['sw_vial']), 0, "sw_vial must be an array of zeros!")
    
    def test_task_4(self):
        global exec_globals # Access the global variable
        exec(self.solutions[3], exec_globals)
        self.assertIsInstance(exec_globals['bridge_indices'], np.ndarray, "bridge_indices must be a NumPy array!")
        self.assertEqual(exec_globals['bridge_indices'].shape, (2,), "bridge_indices must be a 1D array of length 2!")
 
    def test_task_5(self):
        global exec_globals # Access the global variable
        exec(self.solutions[4], exec_globals)
        self.assertIsInstance(exec_globals['translated_signal'], list, "translated_signal must be a list!")
        self.assertEqual(len(exec_globals['translated_signal']), len(exec_globals['signal']), "translated_signal must be the same length as signal!")

    def test_task_6(self):
        global exec_globals # Access the global variable
        exec(self.solutions[5], exec_globals)
        self.assertIsInstance(exec_globals['solution_flask'], np.ndarray, "solution_flask must be a NumPy array!")
        self.assertEqual(exec_globals['solution_flask'].shape, (8, 4), "solution_flask must be a 2D array of shape (8, 4)!")
        self.assertFalse(np.isnan(exec_globals['solution_flask']).any(), "solution_flask should not contain any NaN values!")
        self.assertFalse(np.isinf(exec_globals['solution_flask']).any(), "solution_flask should not contain any infinite values!")
        self.assertTrue((exec_globals['solution_flask'] >= 0).all(), "solution_flask should not contain any negative values!")

    def test_task_7(self):
        global exec_globals # Access the global variable
        exec(self.solutions[6], exec_globals)
        self.assertIsInstance(exec_globals['distilled_solution'], np.float64, "distilled_solution must be a NumPy float64!")
        self.assertEqual(int(exec_globals['distilled_solution']), 24, "distilled_solution must be equal to the answer!")
     


# Main function to run the extraction and tests
def main():
    global exec_globals  # Declare exec_globals as global to modify it
    notebook_path = 'numpyte.ipynb'  # Path to your Jupyter Notebook
    code_cells = extract_code_cells(notebook_path)
    exec_globals = execute_code(code_cells)  # Execute the code cells and store the globals
    unittest.main()  # Run the tests


if __name__ == "__main__":
    main()
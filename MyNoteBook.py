import io, os, sys, types
from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell

path = []
for notebooks in os.listdir():
    if notebooks.endswith('.ipynb'):
        with io.open(notebooks, 'r', encoding='utf-8') as f:
            nb = read(f, 4)
        notebook = notebooks.split('.')[0]
        script = open('%s.py'% notebooks,'w')
        for cell in nb.cells:
            if cell.cell_type == 'code':
                # transform the input to executable Python
                code = InteractiveShell.instance().input_transformer_manager.transform_cell(cell.source)
                script.write(code)
    script.close()
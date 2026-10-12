"""Render report formulae with Matplotlib; independent of document-building runtime."""
import ast,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties

root=Path(__file__).resolve().parent;dest=root/'公式素材';dest.mkdir(exist_ok=True)
tree=ast.parse((root/'build_report.py').read_text(encoding='utf-8'));index=[]
for node in ast.walk(tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='formula':
        latex=ast.literal_eval(node.args[1])
        for piece in latex.split(r'\quad '):
            expression=piece.strip(' ,;')
            name=hashlib.sha256(expression.encode()).hexdigest()[:16]+'.png'
            math_to_image('$'+expression+'$',str(dest/name),prop=FontProperties(size=13),dpi=230,color='black')
            index.append(dict(file=name,latex=expression))
(dest/'公式索引.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
print('Rendered formula parts:',len(index))

from functools import partial

from langchain_core.runnables import RunnableLambda
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

console = Console()
base_style =Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)

def RPrint(preface=""):
    """Simple passthrough "prints, then returns" chain"""
    def print_and_return(x, preface):
        if preface: print(preface, end="")
        pprint(x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def parse_to_history(data):
    """Accepts 'input'/'output' dictionary and return format string"""
    return f"User previously said: {data.get('input')}\nSystem previously responded: {data.get('output')}"
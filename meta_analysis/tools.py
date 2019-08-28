"""Implement some tool function used in other files."""
from colorama import Fore, Style


def print_percent(index=None, total=None,
                  string='',
                  rate=0.1,
                  end='\r',
                  last_end='\n',
                  last_string=' Done',
                  color_last_string=False,
                  flush=True,
                  verbose=True,
                  prefix=None,
                  ):
    """Advanced print."""
    if not verbose:
        return

    if prefix is not None:
        prefix = f'{Style.BRIGHT}[{prefix}]{Style.RESET_ALL} '
        string = prefix+string

    if index is None or total is None:
        print(string, end=last_end)
        return

    period = int(rate*total/100)
    if period == 0 or index % period == 0:
        print(string.format(100*(index+1)/total, index+1, total),
              end=end, flush=flush)

    if color_last_string:
        last_string = f'{Fore.GREEN}{last_string}{Style.RESET_ALL}'

    if index == total-1:
        print((string+last_string).format(100*(index+1)/total, index+1, total),
              end=last_end, flush=flush)

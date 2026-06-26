import re

def main():
    with open('main.tex', 'r', encoding='utf-8') as f:
        text = f.read()

    # Replace **bold** with \textbf{bold}
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
    
    # Replace `code` with \texttt{code}
    text = re.sub(r'`(.*?)`', r'\\texttt{\1}', text)
    
    # Escape ampersands not in equations if any (just specific ones I remember)
    text = text.replace('&\n', '\\&\n').replace(' & ', ' \\& ').replace('& Tinjauan', '\\& Tinjauan')

    with open('main.tex', 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == '__main__':
    main()

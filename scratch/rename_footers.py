import os

templates_dir = "/Users/govind/Desktop/prasad /padma project/Handwritten_Digit_Recognition_Complete/templates"

old_footer_1 = "&copy; 2026 Handwritten Character & Page OCR Recognition Console. MCA Capstone Portfolio."
old_footer_2 = "&copy; 2026 Handwritten Text & Character Recognition System. MCA Final Year Capstone Project."
new_footer = "&copy; 2026 Handwritten Text Recognition System. MCA Capstone Portfolio."

for filename in os.listdir(templates_dir):
    if filename.endswith(".html"):
        filepath = os.path.join(templates_dir, filename)
        with open(filepath, "r") as f:
            content = f.read()
        
        modified = False
        if old_footer_1 in content:
            content = content.replace(old_footer_1, new_footer)
            modified = True
        if old_footer_2 in content:
            content = content.replace(old_footer_2, new_footer)
            modified = True
            
        if modified:
            with open(filepath, "w") as f:
                f.write(content)
            print(f"Updated footer in {filename}")

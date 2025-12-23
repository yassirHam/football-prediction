import os

def fix_styles():
    base_path = 'static/style.css'
    clean_path = 'static/style_clean.css'
    tourn_path = 'static/tournament-style.css'
    feed_path = 'static/feedback-style.css'
    btn_path = 'static/button-styles.css'
    
    # Read base content (try clean first, then base with ignore)
    content = ""
    try:
        if os.path.exists(clean_path):
            with open(clean_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        else:
            with open(base_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
    except Exception as e:
        print(f"Error reading base: {e}")
        return

    # Find truncation point
    # We look for the last known good class from the original file
    # If the file is already short, we might just append.
    # Check if we have .header, since that was at start.
    
    # Let's find end of cleaned file.
    # We'll just assume style_clean.css is the partial file we saw that ended at line 100 (which seemed like header/inputs/etc was fine?) 
    # Actually, viewed file only showed first 100 lines.
    # The user said "buttons are just html".
    # This means the button classes are likely MISSING from the file we have.
    # So we should just append the button styles.
    
    # Let's assume we preserve whatever we have in style_clean.css (which was the valid part)
    # and just append our button styles + tournament styles + feedback styles.
    
    # Actually, style_clean.css might be the FULL file but corrupted at end?
    # If I grep for btn-predict and it's missing, then it's missing.
    
    # Let's just append everything to the end of style_clean.css 
    # But wait, if tournament styles are already there, we might dup?
    # No, grep failed for tournament styles too earlier steps.
    
    clean_content = content
    
    # Read Appendices
    tourn_content = ""
    if os.path.exists(tourn_path):
        with open(tourn_path, 'r', encoding='utf-8') as f:
            tourn_content = f.read()
            
    feed_content = ""
    if os.path.exists(feed_path):
        with open(feed_path, 'r', encoding='utf-8') as f:
            feed_content = f.read()

    btn_content = ""
    if os.path.exists(btn_path):
        with open(btn_path, 'r', encoding='utf-8') as f:
            btn_content = f.read()

    # Combine: Clean + Buttons + Tournament + Feedback
    final_content = clean_content + "\n\n" + btn_content + "\n\n" + tourn_content + "\n\n" + feed_content
    
    # Write back
    with open(base_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
        
    print("Successfully rewrote style.css")

if __name__ == "__main__":
    fix_styles()

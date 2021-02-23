# Coral_Colony_Health_Task2.2

Using image recognition to determine the health of a coral colony by comparing its current condition to past data
Manual page 35-37

===============

perpective.py: basic use of perspective transformation
compare.py: compare coral structures using hardcoded hsv values, contours, and bitwise operations on images
eyedropper.py: can generate masks for coral based on hsv of pixel clicked on, with adjustable +- tolerances

===============

Overall approach:
1. Remove background: eyedropper, hsv masks
2. Alignment: contour? hough lines?		<--- Doing
3. Identify differences

Identifying colors of change:
Growth: new pink branches	Green
Damage: missing branches	Yellow
Bleach: past pink turn white	Red
Recovery: past white turn pink	Blue

Contact me (Oscar) if unclear pls

# Brute Force

 
| @ |  Best Case   |   Worst Case  |
| ------------- | ------------- | ------------- |
|Complexity|     O(n*m)    |     O(n*m)    |
|Explanation|The algorithm still checks all of the possible seams, as every seam must be evaluated, and it energy calculated, therefore the time complexity remains O(rows × cols). | The algorithm evaluates every possible starting column for each seam and processes all rows, resulting in O(rows × cols) complexity regardless of the number of seams removed because it’s constant (50). |
|Example|(Image with uniform energy) |(Full Image Search) |

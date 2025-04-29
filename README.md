# Dynamic

| Case        | Complexity | Explanation                                                                                                                                                                                                                 | Example                                    |
|-------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| Best Case   | O(n\*m)     | The algorithm processes the entire energy matrix to compute the cumulative energy and backtrack to find each seam, requiring operations proportional to the image size (rows × cols) and the number of seams (k).         | Uniform energy distribution in the image   |
| Worst Case  | O(n\*m)     | The algorithm processes the full matrix, regardless of the structure of the energy map or the seam configuration. The number of operations is determined by the size of the image and the number of seams to be removed. | Highly varied energy distribution in image |

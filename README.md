# AlgorithmPro

Image resizing is a fundamental operation in computer vision, graphics processing, and web development. Traditional resizing methods such as scaling and cropping often lead to quality loss or unbalanced aspect ratios. To address this, more intelligent algorithms can be used to preserve important image content while adjusting dimensions.

This project explores three different algorithmic approaches to image resizing:

- Brute Force -> A straightforward method that evaluates all possible resizing paths, ensuring optimal results but at a high computational cost.
- Dynamic Programming –> A more efficient approach that breaks down the problem into smaller subproblems and reuses previous computations to reduce complexity.
- Greedy Algorithm –> A faster but heuristic-based method that makes locally optimal choices at each step, often leading to good (but not always optimal) results.

  
By comparing these techniques, this project aims to analyze their efficiency, accuracy, and suitability for different scenarios. The implementation includes performance benchmarking and visual comparisons of the resized images.

*number of seams=50.

## Brute Force Approach 
![IMG_1726](https://github.com/user-attachments/assets/85c2eb54-1e93-42ac-aade-2d62da0beb9c)


 
| @ |  Best Case   |   Worst Case  |
| ------------- | ------------- | ------------- |
|Complexity|     O(n*m)    |     O(n*m)    |
|Explanation|The algorithm still checks all of the possible seams, as every seam must be evaluated, and it energy calculated, therefore the time complexity remains O(rows × cols). | The algorithm evaluates every possible starting column for each seam and processes all rows, resulting in O(rows × cols) complexity regardless of the number of seams removed because it’s constant (50). |
|Example|(Image with uniform energy) |(Full Image Search) |


## Dynamic Approach 
![IMG_7738](https://github.com/user-attachments/assets/d706c39f-f123-4279-a379-cae6b42b6f21)


| Case        | Complexity | Explanation                                                                                                                                                                                                                 | Example                                    |
|-------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| Best Case   | O(n\*m)     | The algorithm processes the entire energy matrix to compute the cumulative energy and backtrack to find each seam, requiring operations proportional to the image size (rows × cols) and the number of seams (k).         | Uniform energy distribution in the image   |
| Worst Case  | O(n\*m)     | The algorithm processes the full matrix, regardless of the structure of the energy map or the seam configuration. The number of operations is determined by the size of the image and the number of seams to be removed. | Highly varied energy distribution in image |

## Greedy Approach 
![IMG_6563](https://github.com/user-attachments/assets/1be0dbea-65fc-40da-9ad0-4cbbfa1b684c)


| Case        | Complexity | Explanation                                                                                                                                                                                                                 |
|-------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Best Case   | O(n)       | The algorithm processes each of the k seams in O(rows) time, where rows is the number of rows in the image. Since the algorithm only checks the immediate neighbors (left, current, right) at each row, it operates in linear time with respect to the height of the image. |
| Worst Case  | O(n)       | The algorithm processes the image as each seam requires a linear scan through all the rows to select the optimal path. The process is independent of the image width (cols), and each row requires constant-time operations for k seams in O(rows) time.              |


-----
**To wrap it up**
- **Greedy**: Fast but not optimal. It makes locally optimal choices at each step, which can lead to suboptimal global results.
- **Dynamic Programming**: Fast and optimal. It efficiently computes the globally optimal seam using a cumulative energy approach.
- **Brute Force**: Slow but optimal. It guarantees the best result by exploring all possible seams, but it's computationally expensive.


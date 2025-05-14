# Code Review Task

## Objective
Review the following Python code and suggest improvements.

## Code
```python
def process_data(data_list):
    result = []
    for i in range(len(data_list)):
        if data_list[i] > 0:
            result.append(data_list[i] * 2)
        else:
            result.append(0)
    return result
    
# Example usage
data = [5, -3, 10, 0, 8, -2]
processed = process_data(data)
print(processed)
```

## Questions
1. What improvements would you suggest for this code?
2. Are there any potential bugs or edge cases?
3. How would you make this code more efficient?
4. Provide a rewritten version with your improvements.
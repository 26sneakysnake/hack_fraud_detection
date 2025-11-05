# Contributing to Fraud Detection System

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Error logs if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:
- Clear description of the feature
- Use case and benefits
- Potential implementation approach
- Any relevant examples

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/26sneakysnake/hack_fraud_detection.git
   cd hack_fraud_detection
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clear, documented code
   - Follow existing code style
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
   ```bash
   make test
   make lint
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: clear description of your changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Provide clear description of changes
   - Reference any related issues

## 📝 Code Style

### Python
- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to all functions
- Maximum line length: 120 characters

Example:
```python
def calculate_fraud_score(transaction: dict, model: Any) -> float:
    """
    Calculate fraud score for a transaction

    Args:
        transaction: Transaction dictionary with features
        model: Trained fraud detection model

    Returns:
        Fraud probability score (0.0 to 1.0)
    """
    # Implementation here
    pass
```

### Documentation
- Use Markdown for documentation
- Include code examples where helpful
- Keep README.md up to date
- Add inline comments for complex logic

## 🧪 Testing

### Running Tests
```bash
make test
```

### Writing Tests
- Place tests in `tests/` directory
- Use pytest framework
- Aim for high code coverage
- Test edge cases

Example:
```python
def test_load_transactions():
    """Test transaction loading function"""
    df = load_transactions('data/raw/transactions_train.csv')
    assert len(df) > 0
    assert 'transaction_id' in df.columns
    assert 'amount' in df.columns
```

## 📋 Commit Message Guidelines

Use clear, descriptive commit messages:

- **Add**: New feature or functionality
  ```
  Add: user aggregation features
  ```

- **Fix**: Bug fix
  ```
  Fix: handle missing values in card data
  ```

- **Update**: Modify existing feature
  ```
  Update: improve dashboard layout
  ```

- **Refactor**: Code restructuring
  ```
  Refactor: extract feature engineering to separate module
  ```

- **Docs**: Documentation changes
  ```
  Docs: add API documentation
  ```

- **Test**: Add or modify tests
  ```
  Test: add unit tests for data processing
  ```

## 🏗️ Project Structure

When adding new code, maintain the project structure:

```
src/               # Core library code
scripts/           # Executable scripts
notebooks/         # Jupyter notebooks
dashboard/         # Dashboard code
tests/            # Test files
docs/             # Documentation
```

## 🔍 Code Review Process

All submissions require review. We use GitHub Pull Requests for this purpose:

1. Maintainer reviews your PR
2. Feedback and requested changes
3. You update your PR
4. Maintainer approves and merges

## 📦 Dependencies

When adding new dependencies:

1. Add to `requirements.txt`
2. Document why it's needed
3. Check for version compatibility
4. Minimize dependency bloat

## 🐛 Debugging

If you're debugging:

1. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Use print statements sparingly
3. Add unit tests to reproduce bugs
4. Document your findings

## 📚 Resources

- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io)

## 🎯 Priority Areas

Current priority areas for contribution:

1. **Model Improvements**
   - Deep learning models
   - Online learning
   - Model interpretability

2. **Feature Engineering**
   - Geolocation features
   - Network analysis
   - Behavioral sequences

3. **Dashboard Enhancements**
   - Additional visualizations
   - Export functionality
   - User authentication

4. **Documentation**
   - More examples
   - API documentation
   - Tutorial videos

5. **Testing**
   - Increase code coverage
   - Integration tests
   - Performance tests

## ❓ Questions?

If you have questions:
- Open an issue for discussion
- Check existing issues and PRs
- Review project documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## 🙏 Thank You!

Every contribution, no matter how small, is valuable. Thank you for helping improve this project!

---

**Happy Contributing! 🚀**

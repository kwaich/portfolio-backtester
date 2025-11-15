# Implementation Checklist - Code Review Fixes

**Quick Reference**: Track progress on all implementation tasks

---

## Phase 1: Critical & High-Priority ⚡ (Week 1)

### Core Fixes
- [x] **1.1** Cache Expiration System ✅ DONE (Commit: 75f82aa)
  - [x] Add metadata to cache format
  - [x] Implement TTL checking
  - [x] Add CLI argument `--cache-ttl`
  - [x] Update documentation
  - [x] Write 6 new tests

- [x] **1.2** Rate Limiting & Retry Logic ✅ DONE (Commit: baebadc)
  - [x] Create `@retry_with_backoff` decorator
  - [x] Apply to download functions
  - [x] Add delays between batch downloads
  - [x] Write 4 new tests

- [x] **1.3** Import Error Handling ✅ DONE (Commit: 55d1d90)
  - [x] Add try/except for backtest imports
  - [x] Comprehensive error messages for all dependencies
  - [x] N/A - No new tests needed (error handling only)

### Validation
- [🚧] **1.4** Ticker Validation (IN PROGRESS)
  - [ ] Create `validate_ticker()` function
  - [ ] Integrate into backtest.py and app.py
  - [ ] Write 5 new tests

- [ ] **1.5** Date Format Validation
  - [ ] Create `validate_date_string()` function
  - [ ] Update argparse integration
  - [ ] Add date range validation
  - [ ] Write 6 new tests

**Phase 1 Total**: ~12-16 hours, 28 new tests

---

## Phase 2: Code Quality & Organization 🎨 (Week 2)

### Refactoring
- [ ] **2.1** Refactor app.py into Modules
  - [ ] Create `app/` directory structure
  - [ ] Create `config.py`
  - [ ] Create `presets.py`
  - [ ] Create `ui_components.py`
  - [ ] Create `charts.py`
  - [ ] Create `validation.py`
  - [ ] Create `main.py`
  - [ ] Update backward compatibility
  - [ ] Write 7 new tests

- [ ] **2.2** Centralize Session State
  - [ ] Create session state functions
  - [ ] Replace scattered state code
  - [ ] Write 5 new tests

### Cleanup
- [ ] **2.3** Extract Magic Numbers
  - [ ] Add constants to backtest.py
  - [ ] Add constants to app/config.py
  - [ ] Update all references
  - [ ] Write 3 new tests

- [ ] **2.4** Remove Duplicate Code
  - [ ] Extract metric formatting
  - [ ] Extract delta calculations
  - [ ] Write 4 new tests

- [ ] **2.5** Add Logging to plot_backtest.py
  - [ ] Add logging setup
  - [ ] Replace print statements
  - [ ] Write 3 new tests

**Phase 2 Total**: ~16-20 hours, 22 new tests

---

## Phase 3: Performance & Advanced Features ⚡ (Week 3)

### Performance
- [ ] **3.1** Batch Benchmark Downloads
  - [ ] Modify `download_prices()` signature
  - [ ] Update app.py download logic
  - [ ] Update cache for batches
  - [ ] Write 5 new tests

- [ ] **3.2** Minimum Data Validation
  - [ ] Add validation to plot_backtest.py
  - [ ] Add validation to compute_metrics
  - [ ] Add data quality checks
  - [ ] Write 7 new tests

### Advanced (Optional)
- [ ] **3.3** Parallel Downloads
  - [ ] Create `download_prices_parallel()`
  - [ ] Make cache thread-safe
  - [ ] Add CLI option
  - [ ] Write 5 new tests

### Testing
- [ ] **3.4** Integration Tests
  - [ ] Create `test_integration.py`
  - [ ] Add end-to-end workflow tests
  - [ ] Add edge case tests
  - [ ] Add concurrency tests
  - [ ] Add data quality tests
  - [ ] 20+ new integration tests

**Phase 3 Total**: ~12-14 hours, 37+ new tests

---

## Phase 4: Documentation & Polish 📚 (Week 3)

### Documentation
- [ ] **4.1** Update All Documentation
  - [ ] Update README.md
  - [ ] Update CLAUDE.md
  - [ ] Update PROJECT_SUMMARY.md
  - [ ] Create CHANGELOG.md

- [ ] **4.2** Add Deployment Guide
  - [ ] Create DEPLOYMENT.md
  - [ ] Document Streamlit Cloud
  - [ ] Document Docker deployment
  - [ ] Document environment variables

- [ ] **4.3** GitHub Templates
  - [ ] Create bug report template
  - [ ] Create feature request template
  - [ ] Create performance issue template

**Phase 4 Total**: ~6-8 hours

---

## Summary Statistics

### Tasks by Priority
- **High Priority**: 5 tasks (16 hours)
- **Medium Priority**: 6 tasks (20 hours)
- **Low Priority**: 5 tasks (10 hours)

### Testing Goals
- **Current**: 86 tests, 86.1% coverage
- **Target**: 110+ tests, 85%+ coverage
- **New Tests**: 87+ additional tests

### Time Estimate
- **Total Effort**: 40-50 hours
- **Timeline**: 2-3 weeks
- **Team Size**: 1 developer

### Success Metrics
- [ ] All high-priority issues resolved
- [ ] Test coverage ≥ 85%
- [ ] All 110+ tests passing
- [ ] Documentation complete
- [ ] No regressions
- [ ] Performance improved (2x faster downloads)

---

## Daily Progress Tracking

### Week 1
- **Day 1**: ⬜ Task 1.1 (Cache Expiration) - Start
- **Day 2**: ⬜ Task 1.1 (Cache Expiration) - Complete + Tests
- **Day 3**: ⬜ Task 1.2 (Rate Limiting) + Task 1.3 (Import Errors)
- **Day 4**: ⬜ Task 1.4 (Ticker Validation)
- **Day 5**: ⬜ Task 1.5 (Date Validation)
- **Day 6**: ⬜ Phase 1 Testing & Bug Fixes
- **Day 7**: ⬜ Phase 1 Review & Documentation

### Week 2
- **Day 8**: ⬜ Task 2.1 (Refactor) - Structure
- **Day 9**: ⬜ Task 2.1 (Refactor) - Migration
- **Day 10**: ⬜ Task 2.1 (Refactor) - Testing
- **Day 11**: ⬜ Task 2.2 (Session State) + Task 2.3 (Constants)
- **Day 12**: ⬜ Task 2.4 (Duplicate Code) + Task 2.5 (Logging)
- **Day 13**: ⬜ Phase 2 Testing
- **Day 14**: ⬜ Phase 2 Review

### Week 3
- **Day 15**: ⬜ Task 3.1 (Batch Downloads)
- **Day 16**: ⬜ Task 3.2 (Data Validation)
- **Day 17**: ⬜ Task 3.4 (Integration Tests) - Start
- **Day 18**: ⬜ Task 3.4 (Integration Tests) - Complete
- **Day 19**: ⬜ Task 3.3 (Parallel - Optional)
- **Day 20**: ⬜ Phase 4 (Documentation)
- **Day 21**: ⬜ Final Review & Release

---

## Git Workflow

### Branch Strategy
```bash
# Create feature branch
git checkout -b feature/code-review-fixes

# Work on tasks
git add .
git commit -m "feat: implement cache expiration (Task 1.1)"

# Regular pushes
git push -u origin feature/code-review-fixes
```

### Commit Message Convention
- `feat:` - New features
- `fix:` - Bug fixes
- `refactor:` - Code restructuring
- `test:` - Adding tests
- `docs:` - Documentation updates
- `perf:` - Performance improvements

### Example Commits
```
feat: add cache expiration with configurable TTL (Task 1.1)
feat: implement retry logic with exponential backoff (Task 1.2)
fix: add import error handling in app.py (Task 1.3)
refactor: extract app.py into modular structure (Task 2.1)
test: add comprehensive integration test suite (Task 3.4)
docs: update documentation for new features (Task 4.1)
```

---

## Testing Commands

```bash
# Run all tests
pytest -v

# Run specific test file
pytest test_backtest.py -v
pytest test_app.py -v
pytest test_integration.py -v

# Check coverage
pytest --cov=backtest --cov=app --cov-report=term-missing

# Coverage report (HTML)
pytest --cov=backtest --cov=app --cov-report=html
open htmlcov/index.html

# Run only fast tests (exclude slow integration)
pytest -v -m "not slow"

# Run with verbose output
pytest -vv --tb=long
```

---

## Rollback Plan

If any phase causes issues:

1. **Immediate rollback**: `git revert <commit-hash>`
2. **Feature branch isolation**: Keep main branch stable
3. **Incremental merges**: Merge tasks individually
4. **Backup**: Tag before major changes

---

## Notes

- ✅ = Completed
- 🚧 = In Progress
- ⬜ = Not Started
- ❌ = Blocked

Update this checklist as you progress through tasks.

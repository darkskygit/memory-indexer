use crate::MemoryIndex;

use super::FieldIndex;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OptimizeResult {
    pub removed_terms: usize,
    pub removed_keyword_values: usize,
}

impl MemoryIndex {
    pub fn optimize(&mut self) -> OptimizeResult {
        let mut result = OptimizeResult::default();
        for field_index in 0..self.state.fields.len() {
            match &mut self.state.fields[field_index] {
                FieldIndex::Keyword(index) => {
                    let (_, removed) = index.optimize();
                    result.removed_keyword_values += removed;
                }
                FieldIndex::Text(index) => {
                    let (mapping, removed) = index.optimize();
                    result.removed_terms += removed;
                    if removed > 0 {
                        for state in index.doc_states.iter_mut().flatten() {
                            state.terms.retain_mut(|term| {
                                if let Some(new) = mapping[term.term as usize] {
                                    term.term = new;
                                    true
                                } else {
                                    false
                                }
                            });
                            for value in &mut state.values {
                                value.positions.retain_mut(|position| {
                                    if let Some(new) = mapping[position.term as usize] {
                                        position.term = new;
                                        true
                                    } else {
                                        false
                                    }
                                });
                                value.derived.retain_mut(|derived| {
                                    let Some(new_derived) = mapping[derived.derived as usize]
                                    else {
                                        return false;
                                    };
                                    let Some(new_base) = mapping[derived.base as usize] else {
                                        return false;
                                    };
                                    derived.derived = new_derived;
                                    derived.base = new_base;
                                    true
                                });
                            }
                        }
                    }
                }
                FieldIndex::I64(index) => {
                    if let Some(exact) = &mut index.exact {
                        exact.retain(|_, docs| !docs.is_empty());
                    }
                }
                FieldIndex::Bool(_) => {}
            }
        }
        result
    }
}

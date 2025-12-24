use super::{
    ngram::{add_term_to_ngrams, remove_term_from_ngrams, should_index_in_original_aux},
    types::{
        DocData, DomainIndex, DomainLengths, InMemoryIndex, SNAPSHOT_VERSION, SnapshotData,
        TermDomain, TermFrequency, all_domains, domain_config,
    },
};
use std::collections::HashMap;

impl InMemoryIndex {
    pub(super) fn index_maps_mut(&mut self, index_name: &str) -> Index<'_> {
        let key = index_name.to_string();
        Index {
            version: self
                .versions
                .entry(key.clone())
                .or_insert(SNAPSHOT_VERSION)
                .to_owned(),
            docs: self.docs.entry(key.clone()).or_default(),
            domains: self.domains.entry(key.clone()).or_default(),
            total_len: self.total_lens.entry(key.clone()).or_default(),
            domain_total_len: self.domain_total_lens.entry(key).or_default(),
        }
    }

    pub(super) fn index_writer<'a>(
        &'a mut self,
        index_name: &str,
        doc_id: &str,
    ) -> IndexWriter<'a> {
        IndexWriter {
            maps: self.index_maps_mut(index_name),
            doc_id: doc_id.to_string(),
        }
    }
}

pub struct Index<'a> {
    pub(super) version: u32,
    pub(super) docs: &'a mut HashMap<String, DocData>,
    pub(super) domains: &'a mut HashMap<TermDomain, DomainIndex>,
    pub(super) total_len: &'a mut i64,
    pub(super) domain_total_len: &'a mut DomainLengths,
}

impl<'a> Index<'a> {
    fn ensure_domain(&mut self, domain: TermDomain) -> &mut DomainIndex {
        self.domains.entry(domain).or_default()
    }

    pub(super) fn clear(&mut self, only_dicts: bool) {
        if !only_dicts {
            self.docs.clear();
            self.domains.clear();
            *self.total_len = 0;
            self.domain_total_len.clear();
            return;
        }

        for domain_index in self.domains.values_mut() {
            domain_index.term_dict.clear();
            domain_index.ngram_index.clear();
        }
    }

    pub(super) fn import_snapshot(&mut self, snapshot: SnapshotData) {
        let is_legacy = snapshot.version != SNAPSHOT_VERSION;
        self.version = SNAPSHOT_VERSION;

        for (doc_id, mut doc_data) in snapshot.docs {
            if is_legacy {
                normalize_doc_positions(&mut doc_data);
            }
            let domain_lengths = DomainLengths::from_doc(&doc_data);
            doc_data.domain_doc_len = domain_lengths;
            *self.total_len += doc_data.doc_len;
            domain_lengths.for_each_nonzero(|domain, len| {
                self.domain_total_len.add(domain, len);
            });

            if doc_data.term_freqs.is_empty() {
                for (term, positions) in &doc_data.term_pos {
                    let domain_index = self.ensure_domain(TermDomain::Original);
                    let doc_map = domain_index.postings.entry(term.clone()).or_default();
                    doc_map.insert(doc_id.clone(), positions.len() as i64);
                    ensure_aux_for_domain(self, term, TermDomain::Original);
                }
            } else {
                for (term, freqs) in &doc_data.term_freqs {
                    for (domain, count) in freqs.positive_domains() {
                        let domain_index = self.ensure_domain(domain);
                        domain_index
                            .postings
                            .entry(term.clone())
                            .or_default()
                            .insert(doc_id.clone(), count as i64);
                        ensure_aux_for_domain(self, term, domain);
                    }
                }
            }

            self.docs.insert(doc_id, doc_data);
        }

        let mut needs_rebuild = false;
        for domain in all_domains() {
            match snapshot.domains.get(domain) {
                Some(domain_snapshot) => {
                    let has_dict = !domain_snapshot.term_dict.is_empty();
                    let has_ngrams = !domain_config(*domain).enable_ngrams
                        || !domain_snapshot.ngram_index.is_empty();
                    if has_dict && has_ngrams {
                        let domain_index = self.ensure_domain(*domain);
                        domain_index.term_dict = domain_snapshot.term_dict.clone();
                        domain_index.ngram_index = domain_snapshot.ngram_index.clone();
                    } else {
                        needs_rebuild = true;
                    }
                }
                None => needs_rebuild = true,
            }
        }

        if needs_rebuild {
            self.rebuild_aux_indices();
        }
    }

    fn rebuild_aux_indices(&mut self) {
        if self.docs.is_empty() {
            return;
        }

        for domain_index in self.domains.values_mut() {
            domain_index.term_dict.clear();
            domain_index.ngram_index.clear();
        }

        let mut tasks: Vec<(String, Vec<TermDomain>)> = Vec::new();
        for doc_data in self.docs.values() {
            if doc_data.term_freqs.is_empty() {
                let domains = vec![TermDomain::Original];
                for term in doc_data.term_pos.keys() {
                    tasks.push((term.clone(), domains.clone()));
                }
                continue;
            }

            for (term, freqs) in &doc_data.term_freqs {
                let domains: Vec<TermDomain> = freqs
                    .positive_domains()
                    .into_iter()
                    .map(|(domain, _)| domain)
                    .collect();
                if !domains.is_empty() {
                    tasks.push((term.clone(), domains));
                }
            }
        }

        for (term, domains) in tasks {
            for domain in domains {
                ensure_aux_for_domain(self, &term, domain);
            }
        }
    }

    pub fn ensure_aux_indices_for_doc(&mut self, term_freqs: &HashMap<String, TermFrequency>) {
        for (term, freqs) in term_freqs {
            for (domain, _) in freqs.positive_domains() {
                ensure_aux_for_domain(self, term, domain);
            }
        }
    }

    pub fn remove_doc_terms(&mut self, doc_id: &str, doc_data: &DocData) {
        let mut remove_terms: Vec<(String, TermDomain)> = Vec::new();

        let mut process_term = |term: &str, domain: TermDomain, maps: &mut Index<'_>| {
            if let Some(domain_index) = maps.domains.get_mut(&domain) {
                if let Some(doc_map) = domain_index.postings.get_mut(term) {
                    doc_map.remove(doc_id);
                    if doc_map.is_empty() {
                        remove_terms.push((term.to_string(), domain));
                    }
                }
            }
        };

        if doc_data.term_freqs.is_empty() {
            for term in doc_data.term_pos.keys() {
                process_term(term, TermDomain::Original, self);
            }
        } else {
            for (term, freqs) in &doc_data.term_freqs {
                for (domain, _) in freqs.positive_domains() {
                    process_term(term, domain, self);
                }
            }
        }

        for (term, domain) in remove_terms {
            if let Some(domain_index) = self.domains.get_mut(&domain) {
                let should_remove_term = domain_index
                    .postings
                    .get(&term)
                    .map(|docs| docs.is_empty())
                    .unwrap_or(true);
                if should_remove_term {
                    domain_index.postings.remove(&term);
                    remove_term_from_aux(domain_index, domain, &term);
                }
            }
        }
    }
}

impl From<Index<'_>> for Option<SnapshotData> {
    fn from(maps: Index<'_>) -> Self {
        if maps.docs.is_empty() {
            return None;
        }

        let mut domains = HashMap::new();
        for (domain, domain_index) in maps.domains.iter() {
            domains.insert(
                *domain,
                super::types::DomainSnapshot {
                    term_dict: domain_index.term_dict.clone(),
                    ngram_index: domain_index.ngram_index.clone(),
                },
            );
        }

        Some(SnapshotData {
            version: maps.version,
            docs: maps.docs.clone(),
            domains,
        })
    }
}

pub struct IndexWriter<'a> {
    pub(super) maps: Index<'a>,
    doc_id: String,
}

impl<'a> IndexWriter<'a> {
    pub fn add_term_frequency(&mut self, term: &str, freqs: &TermFrequency) {
        for (domain, count) in freqs.positive_domains() {
            self.add_term(term, domain, count as i64);
        }
    }

    pub fn add_term(&mut self, term: &str, domain: TermDomain, freq: i64) {
        if freq <= 0 {
            return;
        }

        let domain_index = self.maps.ensure_domain(domain);
        domain_index
            .postings
            .entry(term.to_string())
            .or_default()
            .insert(self.doc_id.clone(), freq);

        ensure_aux_for_domain(&mut self.maps, term, domain);
    }
}

fn ensure_aux_for_domain(maps: &mut Index<'_>, term: &str, domain: TermDomain) {
    if !should_index_in_domain_dict(domain, term) {
        return;
    }

    let config = domain_config(domain);
    let domain_index = maps.ensure_domain(domain);
    domain_index.term_dict.insert(term.to_string());
    if config.enable_ngrams {
        add_term_to_ngrams(&mut domain_index.ngram_index, term);
    }
}

fn remove_term_from_aux(domain_index: &mut DomainIndex, domain: TermDomain, term: &str) {
    domain_index.term_dict.remove(term);
    if domain_config(domain).enable_ngrams {
        remove_term_from_ngrams(&mut domain_index.ngram_index, term);
    }
}

fn should_index_in_domain_dict(domain: TermDomain, term: &str) -> bool {
    match domain {
        TermDomain::Original => should_index_in_original_aux(term),
        _ => true,
    }
}

fn normalize_doc_positions(doc_data: &mut DocData) {
    // Legacy snapshots stored derived-domain spans in term_pos; migrate them into
    // derived_terms so they stay aligned once offsets are normalized.
    if doc_data.term_freqs.is_empty() {
        doc_data.derived_terms.clear();
        return;
    }

    let mut derived: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
    doc_data.term_pos.retain(|term, positions| {
        let is_original = doc_data
            .term_freqs
            .get(term)
            .map(|f| f.get(TermDomain::Original) > 0)
            .unwrap_or(true);
        if is_original {
            true
        } else {
            for (start, end) in positions {
                if let (Some(s), Some(e)) =
                    (usize::try_from(*start).ok(), usize::try_from(*end).ok())
                {
                    if s <= e && e <= doc_data.content.len() {
                        derived
                            .entry(term.clone())
                            .or_default()
                            .push((s as u32, e as u32));
                    }
                }
            }
            false
        }
    });

    if doc_data.derived_terms.is_empty() && !derived.is_empty() {
        for spans in derived.values_mut() {
            spans.sort();
            spans.dedup();
            if let Some(min_len) = spans.iter().map(|(s, e)| e - s).min() {
                spans.retain(|(s, e)| e - s == min_len);
            }
        }
        doc_data.derived_terms = derived;
    }
}

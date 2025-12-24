use super::{
    ngram::{add_term_to_ngrams, remove_term_from_ngrams, should_index_in_original_aux},
    types::{
        DocData, DomainIndex, DomainLengths, InMemoryIndex, SNAPSHOT_VERSION, SnapshotData,
        TermDomain, TermFrequency, domain_config,
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
        self.version = snapshot.version;
        *self.total_len = snapshot.total_len;
        *self.domain_total_len = snapshot.domain_total_len;
        *self.docs = snapshot.docs;
        *self.domains = snapshot.domains;

        // Drop aux structures for domains that never run fuzzy/aux lookups to reduce memory.
        for (domain, index) in self.domains.iter_mut() {
            let config = domain_config(*domain);
            if !config.allow_fuzzy && !config.enable_ngrams {
                index.term_dict.clear();
                index.ngram_index.clear();
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

        Some(SnapshotData {
            version: maps.version,
            docs: maps.docs.clone(),
            domains: maps.domains.clone(),
            total_len: *maps.total_len,
            domain_total_len: *maps.domain_total_len,
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
    let config = domain_config(domain);
    if !config.allow_fuzzy && !config.enable_ngrams {
        // Prefix domains don't participate in fuzzy search; skip aux structures to save memory.
        return;
    }

    if !should_index_in_domain_dict(domain, term) {
        return;
    }

    let domain_index = maps.ensure_domain(domain);
    if config.allow_fuzzy {
        domain_index.term_dict.insert(term.to_string());
    }
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

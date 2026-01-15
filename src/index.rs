use super::types::{
    DocData, DocId, DomainIndex, InMemoryIndex, IndexState, Posting, SNAPSHOT_VERSION,
    SnapshotData, TermDomain, TermId,
};

impl InMemoryIndex {
    pub(super) fn index_state_mut(&mut self, index_name: &str) -> &mut IndexState {
        self.indexes
            .entry(index_name.to_string())
            .or_insert_with(|| IndexState {
                version: SNAPSHOT_VERSION,
                ..Default::default()
            })
    }
}

pub struct Index<'a> {
    pub(super) state: &'a mut IndexState,
}

impl<'a> Index<'a> {
    fn ensure_domain(&mut self, domain: TermDomain) -> &mut DomainIndex {
        self.state.domains.entry(domain).or_default()
    }

    pub(super) fn clear(&mut self) {
        *self.state = IndexState::default();
    }

    pub(super) fn import_snapshot(&mut self, snapshot: SnapshotData) {
        self.state.version = snapshot.version;
        self.state.terms = snapshot.terms;
        self.state.docs = snapshot.docs;
        self.state.doc_ids = snapshot.doc_ids;
        self.state.domains = snapshot.domains;
        self.state.total_len = snapshot.total_len;
        self.state.domain_total_len = snapshot.domain_total_len;

        self.rebuild_indexes();
    }

    fn rebuild_indexes(&mut self) {
        self.state.term_index.clear();
        self.state.term_index.reserve(self.state.terms.len());
        for (idx, term) in self.state.terms.iter().enumerate() {
            self.state.term_index.insert(term.clone(), idx as TermId);
        }

        self.state.doc_index.clear();
        self.state.free_docs.clear();
        for (idx, doc_id) in self.state.doc_ids.iter().enumerate() {
            match self.state.docs.get(idx).and_then(|doc| doc.as_ref()) {
                Some(_) => {
                    self.state.doc_index.insert(doc_id.clone(), idx as DocId);
                }
                None => self.state.free_docs.push(idx as DocId),
            }
        }

        self.state.dirty.clear();
        self.state.deleted.clear();
    }

    pub(super) fn add_posting(&mut self, term: TermId, domain: TermDomain, doc: DocId, freq: u32) {
        if freq == 0 {
            return;
        }
        let domain_index = self.ensure_domain(domain);
        let postings = domain_index.postings.entry(term).or_default();
        if let Some(existing) = postings.iter_mut().find(|p| p.doc == doc) {
            existing.freq += freq;
        } else {
            postings.push(Posting { doc, freq });
        }
        if let Ok(mut aux) = domain_index.aux.write() {
            aux.clear();
        }
    }

    pub(super) fn remove_doc_terms(&mut self, doc_id: DocId, doc_data: &DocData) {
        let mut remove_terms: Vec<(TermId, TermDomain)> = Vec::new();

        let mut process_term = |term: TermId, domain: TermDomain, maps: &mut Index<'_>| {
            if let Some(domain_index) = maps.state.domains.get_mut(&domain)
                && let Some(postings) = domain_index.postings.get_mut(&term)
            {
                let before = postings.len();
                postings.retain(|p| p.doc != doc_id);
                if postings.len() != before
                    && let Ok(mut aux) = domain_index.aux.write()
                {
                    aux.clear();
                }
                if postings.is_empty() {
                    remove_terms.push((term, domain));
                }
            }
        };

        for entry in &doc_data.term_freqs {
            for (domain, _) in entry.positive_domains() {
                process_term(entry.term, domain, self);
            }
        }

        for (term, domain) in remove_terms {
            if let Some(domain_index) = self.state.domains.get_mut(&domain) {
                let should_remove = domain_index
                    .postings
                    .get(&term)
                    .map(|docs| docs.is_empty())
                    .unwrap_or(true);
                if should_remove {
                    domain_index.postings.remove(&term);
                    if let Ok(mut aux) = domain_index.aux.write() {
                        aux.clear();
                    }
                }
            }
        }
    }
}

impl From<Index<'_>> for Option<SnapshotData> {
    fn from(maps: Index<'_>) -> Self {
        if maps.state.docs.iter().all(|d| d.is_none()) {
            return None;
        }

        Some(SnapshotData {
            version: maps.state.version,
            terms: maps.state.terms.clone(),
            docs: maps.state.docs.clone(),
            doc_ids: maps.state.doc_ids.clone(),
            domains: maps.state.domains.clone(),
            total_len: maps.state.total_len,
            domain_total_len: maps.state.domain_total_len,
        })
    }
}

// BioVanguard - Split-Pane Dashboard Application Logic

class BioVanguardApp {
    constructor() {
        this.selectedFile = null;
        this.selectedExample = null;
        this.currentResult = null;
        this.sdfContent = '';
        this.viewer = null;
        this.fallbackViewer = null;
        this.residueLabels = [];
        this.currentResidues = [];
        this.allResidues = [];
        this.labelsVisible = true;
        this.surfaceVisible = false;
        this.surfaceId = null;
        this.currentLigandStyle = 'stick';
        this.visualResidueLimit = 14;
        this.apiBaseUrl = this.resolveApiBaseUrl();
        this.selectedResidueIndex = null;
        this.selectedResidue = null;
        this.init();
    }

    resolveApiBaseUrl() {
        const { protocol, hostname, port, origin } = window.location;
        const localHostnames = new Set(['', 'localhost', '127.0.0.1', '::1']);

        if (protocol === 'file:' || origin === 'null') {
            return 'http://127.0.0.1:8000';
        }

        if (localHostnames.has(hostname) && port !== '8000') {
            return 'http://127.0.0.1:8000';
        }

        return origin;
    }

    init() {
        this.setupEventListeners();
        this.loadExampleLigands();
    }

    async ensureBackendAvailable() {
        const controller = new AbortController();
        const timeoutId = window.setTimeout(() => controller.abort(), 6000);

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/health`, {
                cache: 'no-store',
                signal: controller.signal
            });

            if (!response.ok) {
                throw new Error(`Health check returned ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            throw new Error(`Backend is not reachable at ${this.apiBaseUrl}. Start the local server on port 8000 and try again.`);
        } finally {
            window.clearTimeout(timeoutId);
        }
    }

    setupEventListeners() {
        // Drop zone interactions
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');

        dropZone.addEventListener('click', () => fileInput.click());
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });

        // Clear button
        // Detail panel event listeners
        document.getElementById('closeDetailBtn')?.addEventListener('click', () => {
            this.hideInteractionDetail();
        });

        document.getElementById('zoomToResidueBtn')?.addEventListener('click', () => {
            if (this.selectedResidue) {
                this.focusResidue(this.selectedResidue);
            }
        });

        document.getElementById('copyResidueInfoBtn')?.addEventListener('click', () => {
            this.copyResidueInfo();
        });

        document.getElementById('clearBtn').addEventListener('click', () => {
            this.clearFile();
        });

        // Copy SDF content
        document.getElementById('copyBtn').addEventListener('click', () => {
            this.copySdfContent();
        });

        // Download SDF
        document.getElementById('downloadSdfBtn').addEventListener('click', () => {
            this.downloadSdf();
        });

        // Predict button
        document.getElementById('predictBtn').addEventListener('click', () => {
            this.runPrediction();
        });

        // Viewer controls
        document.getElementById('resetViewBtn')?.addEventListener('click', () => {
            this.resetView();
        });

        document.getElementById('toggleLabelsBtn')?.addEventListener('click', () => {
            this.toggleLabels();
        });

        document.getElementById('toggleSurfaceBtn')?.addEventListener('click', () => {
            this.toggleSurface();
        });

        // New view mode buttons
        document.getElementById('ligandFocusBtn')?.addEventListener('click', () => {
            this.focusOnLigand();
        });

        document.getElementById('bindingPocketBtn')?.addEventListener('click', () => {
            this.showBindingPocket();
        });

        document.getElementById('interactionMapBtn')?.addEventListener('click', () => {
            this.showInteractionMap();
        });

        document.getElementById('styleSelect')?.addEventListener('change', (e) => {
            this.changeStyle(e.target.value);
        });

        // Download buttons
        document.getElementById('downloadPDF').addEventListener('click', () => {
            this.downloadReport('pdf');
        });

        document.getElementById('downloadJSON').addEventListener('click', () => {
            this.downloadReport('json');
        });
    }

    loadExampleLigands() {
        const examples = [
            { name: 'Example 1', id: 'ZINC000000000171', file: 'ZINC000000000171.sdf' },
            { name: 'Example 2', id: 'ZINC000000000561', file: 'ZINC000000000561.sdf' },
            { name: 'Example 3', id: 'ZINC000002622269', file: 'ZINC000002622269.sdf' },
            { name: 'Example 4', id: 'ZINC000015675944', file: 'ZINC000015675944.sdf' }
        ];

        const grid = document.getElementById('exampleGrid');
        grid.innerHTML = examples.map(ex => `
            <div class="example-item" data-file="${ex.file}">
                <div class="example-name">${ex.name}</div>
                <div class="example-id">${ex.id}</div>
            </div>
        `).join('');

        // Add click handlers
        document.querySelectorAll('.example-item').forEach(item => {
            item.addEventListener('click', () => {
                this.selectExample(item);
            });
        });
    }

    async handleFileSelect(file) {
        if (!file.name.endsWith('.sdf')) {
            this.showAlert('Please select a valid SDF file', 'error');
            return;
        }

        this.selectedFile = file;
        this.selectedExample = null;
        
        // Clear example selection
        document.querySelectorAll('.example-item').forEach(item => {
            item.classList.remove('selected');
        });

        // Read file content
        try {
            const content = await this.readFileContent(file);
            this.sdfContent = content;
            this.displaySdfContent(content);
            
            // Update UI
            document.getElementById('fileName').textContent = file.name;
            document.getElementById('fileMeta').textContent = this.formatFileSize(file.size);
            document.getElementById('fileInfoPanel').classList.remove('hidden');
            document.getElementById('predictBtn').disabled = false;

            this.showAlert(`File "${file.name}" loaded successfully`, 'success');
        } catch (error) {
            this.showAlert(`Error reading file: ${error.message}`, 'error');
        }
    }

    async selectExample(element) {
        // Clear file selection
        this.selectedFile = null;
        document.getElementById('fileInput').value = '';
        document.getElementById('fileInfoPanel').classList.add('hidden');

        // Update selection
        document.querySelectorAll('.example-item').forEach(item => {
            item.classList.remove('selected');
        });
        element.classList.add('selected');

        this.selectedExample = element.dataset.file;
        
        // Load example SDF content
        try {
            const response = await fetch(`${this.apiBaseUrl}/PredictionPipeline/${this.selectedExample}`);
            if (response.ok) {
                const content = await response.text();
                this.sdfContent = content;
                this.displaySdfContent(content);
            }
        } catch (error) {
            console.error('Error loading example:', error);
        }

        document.getElementById('predictBtn').disabled = false;

        const exampleId = element.querySelector('.example-id').textContent;
        this.showAlert(`Example ligand "${exampleId}" selected`, 'success');
    }

    readFileContent(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    displaySdfContent(content) {
        const sdfWindow = document.getElementById('sdfContent');
        
        // Format content with line numbers
        const lines = content.split('\n');
        const formattedContent = lines.map((line, index) => {
            const lineNum = String(index + 1).padStart(4, ' ');
            return `<span style="color: #666;">${lineNum}</span> ${this.escapeHtml(line)}`;
        }).join('\n');
        
        sdfWindow.innerHTML = `<pre>${formattedContent}</pre>`;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    clearFile() {
        this.selectedFile = null;
        this.sdfContent = '';
        document.getElementById('fileInput').value = '';
        document.getElementById('fileInfoPanel').classList.add('hidden');
        document.getElementById('predictBtn').disabled = true;
        
        // Clear SDF content display
        const sdfWindow = document.getElementById('sdfContent');
        sdfWindow.innerHTML = `
            <div class="placeholder-text">
                <div class="placeholder-icon">3D</div>
                <p>SDF Coordinate Data Will Appear Here</p>
                <p class="placeholder-hint">Upload Or Drop An SDF File To View Molecular Structure Data</p>
            </div>
        `;
        
        this.showAlert('File cleared', 'info');
    }

    copySdfContent() {
        if (!this.sdfContent) {
            this.showAlert('No content to copy', 'error');
            return;
        }

        navigator.clipboard.writeText(this.sdfContent).then(() => {
            this.showAlert('SDF content copied to clipboard', 'success');
        }).catch(() => {
            this.showAlert('Failed to copy content', 'error');
        });
    }

    downloadSdf() {
        if (!this.sdfContent) {
            this.showAlert('No content to download', 'error');
            return;
        }

        const blob = new Blob([this.sdfContent], { type: 'chemical/x-mdl-sdfile' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = this.selectedFile ? this.selectedFile.name : 'ligand.sdf';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        this.showAlert('SDF file downloaded', 'success');
    }

    async runPrediction() {
        if (!this.selectedFile && !this.selectedExample) {
            this.showAlert('Please select a ligand file or example', 'error');
            return;
        }

        // Show progress overlay
        document.getElementById('progressOverlay').classList.remove('hidden');

        try {
            // Prepare form data
            const formData = new FormData();
            if (this.selectedFile) {
                formData.append('file', this.selectedFile);
            } else {
                formData.append('example', this.selectedExample);
            }

            // Update progress
            this.updateProgress(0, 'Checking backend...');
            await this.ensureBackendAvailable();
            this.updateProgress(10, 'Uploading ligand...');
            await this.sleep(300);

            // Send prediction request
            this.updateProgress(25, 'Building molecular graph...');
            const response = await fetch(`${this.apiBaseUrl}/api/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                let detail = response.statusText;
                try {
                    const errorPayload = await response.json();
                    detail = errorPayload.error || detail;
                } catch (_) {
                    // Keep the HTTP status text when the response is not JSON.
                }
                throw new Error(`Server error: ${detail}`);
            }

            this.updateProgress(50, 'Running GNN prediction...');
            await this.sleep(1000);

            this.updateProgress(75, 'Computing XGBoost features...');
            const result = await response.json();
            await this.sleep(500);

            this.updateProgress(90, 'Retrieving binding pose...');
            await this.sleep(500);

            this.updateProgress(100, 'Complete!');
            await this.sleep(500);

            // Hide progress and display results
            document.getElementById('progressOverlay').classList.add('hidden');
            this.displayResults(result);

        } catch (error) {
            console.error('Prediction error:', error);
            document.getElementById('progressOverlay').classList.add('hidden');
            this.showAlert(`Prediction failed: ${error.message}`, 'error');
        }
    }

    displayResults(result) {
        // Show results panel
        document.getElementById('resultsPanel').classList.remove('hidden');
        document.getElementById('viewerControls').style.display = 'flex';

        // Ensemble score
        const ensembleScore = result.ensemble_score;
        const isActive = ensembleScore <= -7.0;
        
        const ensembleEl = document.getElementById('ensembleScore');
        ensembleEl.textContent = ensembleScore.toFixed(3);
        ensembleEl.className = `result-value ${isActive ? 'active' : 'inactive'}`;
        
        const statusEl = document.getElementById('activityStatus');
        statusEl.textContent = isActive ? 'ACTIVE' : 'INACTIVE';
        statusEl.className = `result-status ${isActive ? 'active' : 'inactive'}`;

        // Component scores
        if (result.gnn_pred !== null) {
            document.getElementById('gnnScore').textContent = result.gnn_pred.toFixed(3);
        }
        if (result.xgb_pred !== null) {
            document.getElementById('xgbScore').textContent = result.xgb_pred.toFixed(3);
        }

        // Molecular properties
        const props = result.properties;
        document.getElementById('molWeight').textContent = props.mw.toFixed(2);
        document.getElementById('logP').textContent = props.logp.toFixed(2);
        document.getElementById('hbd').textContent = props.hbd;
        document.getElementById('hba').textContent = props.hba;
        document.getElementById('tpsa').textContent = props.tpsa.toFixed(2);
        document.getElementById('qed').textContent = props.qed.toFixed(3);

        this.displayModelDiagnostics(result);
        this.displayBindingResidues(result);
        this.displayDrugLikeness(result.drug_likeness);

        // Store result for downloads
        this.currentResult = result;

        // Initialize 3D viewer
        this.init3DViewer(result);

        this.showAlert('Prediction completed successfully!', 'success');
    }

    displayModelDiagnostics(result) {
        const diagnostics = result.model_diagnostics || {};
        const residuals = diagnostics.residuals || {};
        const weights = result.model_stack?.ensemble_weights || {};

        const ensembleMeta = document.getElementById('ensembleMeta');
        if (ensembleMeta) {
            const gnnWeight = weights.gnn !== undefined ? `${Math.round(weights.gnn * 100)}%` : '60%';
            const xgbWeight = weights.xgboost !== undefined ? `${Math.round(weights.xgboost * 100)}%` : '40%';
            ensembleMeta.textContent = `Best model ensemble: ${gnnWeight} GNN / ${xgbWeight} XGBoost`;
        }

        this.setText('agreementScore', diagnostics.agreement_score !== undefined ? `${Math.round(diagnostics.agreement_score * 100)}%` : '--');
        this.setText('modelSpread', diagnostics.model_spread !== undefined ? diagnostics.model_spread.toFixed(3) : '--');
        this.setText('uncertaintyBand', diagnostics.uncertainty_band !== undefined ? `+/-${diagnostics.uncertainty_band.toFixed(3)}` : '--');
        this.setText('gnnResidual', residuals.gnn_to_ensemble !== null && residuals.gnn_to_ensemble !== undefined ? residuals.gnn_to_ensemble.toFixed(3) : '--');
        this.setText('xgbResidual', residuals.xgboost_to_ensemble !== undefined ? residuals.xgboost_to_ensemble.toFixed(3) : '--');
        this.setText('modelInterpretation', diagnostics.interpretation || '--');

        const pose = result.pose_meta || {};
        this.setText('poseSource', pose.source_file || 'original');
        this.setText('poseSimilarity', pose.similarity !== undefined ? Number(pose.similarity).toFixed(3) : '--');
        this.setText('poseRmsd', pose.align_rmsd !== undefined ? Number(pose.align_rmsd).toFixed(2) : '--');
    }

    getVisualResidues(residues) {
        return (residues || []).slice(0, this.visualResidueLimit);
    }

    displayBindingResidues(result) {
        const allResidues = result.binding_residues || [];
        this.allResidues = allResidues;
        const residues = this.getVisualResidues(allResidues);
        const countEl = document.getElementById('residueCount');
        if (countEl) {
            const cutoff = result.binding_cutoff_angstrom || 4.5;
            countEl.textContent = allResidues.length > residues.length
                ? `${residues.length} of ${allResidues.length} residues visualized within ${cutoff} Angstrom`
                : `${residues.length} residues visualized within ${cutoff} Angstrom`;
        }

        const list = document.getElementById('residueList');
        if (!list) return;

        if (!residues.length) {
            list.innerHTML = '<div class="empty-state">No binding residues found for this pose.</div>';
            return;
        }

        // Helper function to get interaction icon and color
        const getInteractionBadge = (interaction) => {
            const badges = {
                'H-Bond': { color: '#4CAF50', label: 'H-Bond' },
                'Hydrophobic': { color: '#FF9800', label: 'Hydrophobic' },
                'Pi-Stacking': { color: '#9C27B0', label: 'Pi-Stack' },
                'Salt Bridge': { color: '#F44336', label: 'Salt Bridge' },
                'Van Der Waals': { color: '#607D8B', label: 'vdW' }
            };
            return badges[interaction] || { color: '#999', label: interaction };
        };

        list.innerHTML = residues.map((residue, index) => {
            const interactions = residue.interactions || [];
            const interactionBadges = interactions.map(int => {
                const badge = getInteractionBadge(int);
                return `<span class="interaction-badge" style="background-color: ${badge.color}20; color: ${badge.color}; border-color: ${badge.color}40;" title="${badge.label}">
                    ${badge.label}
                </span>`;
            }).join('');

            return `
                <button class="residue-row" data-index="${index}" type="button">
                    <div class="residue-info">
                        <span class="residue-name">${this.escapeHtml(residue.label || residue.res_name)}</span>
                        <span class="residue-distance">${Number(residue.min_distance).toFixed(2)} A</span>
                    </div>
                    <div class="residue-interactions">
                        ${interactionBadges || '<span class="interaction-badge-empty">No specific interactions</span>'}
                    </div>
                </button>
            `;
        }).join('');

        list.querySelectorAll('.residue-row').forEach(row => {
            row.addEventListener('click', () => {
                const index = Number(row.dataset.index);
                const residue = residues[index];
                this.showInteractionDetail(residue, index);
                this.focusResidue(residue);
            });
        });
    }

    showInteractionDetail(residue, index) {
        this.selectedResidueIndex = index;
        this.selectedResidue = residue;
        const panel = document.getElementById('interactionDetailPanel');
        if (!panel) return;

        // Helper function to get interaction badge
        const getInteractionBadge = (interaction) => {
            const badges = {
                'H-Bond': { color: '#4CAF50', label: 'H-Bond' },
                'Hydrophobic': { color: '#FF9800', label: 'Hydrophobic' },
                'Pi-Stacking': { color: '#9C27B0', label: 'Pi-Stack' },
                'Salt Bridge': { color: '#F44336', label: 'Salt Bridge' },
                'Van Der Waals': { color: '#607D8B', label: 'vdW' }
            };
            return badges[interaction] || { color: '#999', label: interaction };
        };

        // Update panel title
        document.getElementById('detailPanelTitle').textContent = `${residue.label || residue.res_name} Interactions`;

        // Update residue info
        document.getElementById('detailResidueName').textContent = residue.label || residue.res_name || 'Unknown';
        document.getElementById('detailResidueChain').textContent = `Chain: ${residue.chain || '-'}`;
        document.getElementById('detailResidueId').textContent = `ID: ${residue.res_id || '-'}`;
        document.getElementById('detailResidueDistance').textContent = `Distance: ${Number(residue.min_distance).toFixed(2)} Angstrom`;

        // Update interaction badges
        const interactions = residue.interactions || [];
        const badgesContainer = document.getElementById('detailInteractionBadges');
        if (interactions.length > 0) {
            badgesContainer.innerHTML = interactions.map(int => {
                const badge = getInteractionBadge(int);
                return `<span class="interaction-badge" style="background-color: ${badge.color}20; color: ${badge.color}; border-color: ${badge.color}40;">
                    ${badge.label}
                </span>`;
            }).join('');
        } else {
            badgesContainer.innerHTML = '<span class="detail-empty">No specific interactions detected</span>';
        }

        // Update contacts list
        const contacts = residue.contacts || [];
        const contactsContainer = document.getElementById('detailContactsList');
        const contactCount = document.getElementById('detailContactCount');
        contactCount.textContent = `${contacts.length} contact${contacts.length !== 1 ? 's' : ''}`;

        if (contacts.length > 0) {
            contactsContainer.innerHTML = contacts.map(contact => {
                const types = contact.types || contact.interaction_types || [];
                const typeBadges = types.map(type => {
                    return `<span class="detail-contact-type-badge">${type}</span>`;
                }).join('');

                return `
                    <div class="detail-contact-item">
                        <div class="detail-contact-header">
                            <div class="detail-contact-atoms">
                                ${this.escapeHtml(contact.lig_atom || 'Ligand')} to ${this.escapeHtml(contact.prot_atom || 'Protein')}
                            </div>
                            <div class="detail-contact-distance">
                                ${Number(contact.distance).toFixed(2)} Angstrom
                            </div>
                        </div>
                        ${typeBadges ? `<div class="detail-contact-types">${typeBadges}</div>` : ''}
                    </div>
                `;
            }).join('');
        } else {
            contactsContainer.innerHTML = '<div class="detail-empty" style="padding: 1rem;">No contact details available</div>';
        }

        // Show panel
        panel.classList.remove('hidden');
        
        // Scroll to panel
        setTimeout(() => {
            panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }

    hideInteractionDetail() {
        const panel = document.getElementById('interactionDetailPanel');
        if (panel) {
            panel.classList.add('hidden');
        }
        this.selectedResidueIndex = null;
        this.selectedResidue = null;
    }

    copyResidueInfo() {
        if (!this.selectedResidue) {
            this.showAlert('No residue selected', 'error');
            return;
        }

        const residue = this.selectedResidue;
        const interactions = residue.interactions || [];
        const contacts = residue.contacts || [];

        let info = `Residue: ${residue.label || residue.res_name}\n`;
        info += `Chain: ${residue.chain || '-'}\n`;
        info += `ID: ${residue.res_id || '-'}\n`;
        info += `Distance: ${Number(residue.min_distance).toFixed(2)} Angstrom\n\n`;
        
        if (interactions.length > 0) {
            info += `Interactions: ${interactions.join(', ')}\n\n`;
        }

        if (contacts.length > 0) {
            info += `Contacts (${contacts.length}):\n`;
            contacts.forEach((contact, i) => {
                info += `${i + 1}. ${contact.lig_atom || 'Ligand'} to ${contact.prot_atom || 'Protein'}: ${Number(contact.distance).toFixed(2)} Angstrom`;
                const types = contact.types || contact.interaction_types || [];
                if (types.length > 0) {
                    info += ` [${types.join(', ')}]`;
                }
                info += '\n';
            });
        }

        navigator.clipboard.writeText(info).then(() => {
            this.showAlert('Residue information copied to clipboard', 'success');
        }).catch(() => {
            this.showAlert('Failed to copy to clipboard', 'error');
        });
    }

    addViewerResidueBadge(viewerElement, residueCount) {
        const existing = viewerElement.querySelector('.viewer-count-badge');
        if (existing) {
            existing.remove();
        }

        const badge = document.createElement('div');
        badge.className = 'viewer-count-badge';
        badge.textContent = `${residueCount} Binding Residues Visualized`;
        viewerElement.appendChild(badge);
    }

    init3DViewer(result) {
        const viewerElement = document.getElementById('viewer3d');
        viewerElement.innerHTML = '';
        viewerElement.closest('.viewer-3d-container')?.classList.add('has-result');

        this.viewer = null;
        this.residueLabels = [];
        this.labelsVisible = true;
        this.surfaceVisible = false;
        this.surfaceId = null;
        this.currentResidues = this.getVisualResidues(result.binding_residues || []);
        this.fallbackViewer = null;
        this.currentLigandStyle = document.getElementById('styleSelect')?.value || this.currentLigandStyle || 'stick';
        this.updateViewerControlState();

        if (!window.$3Dmol) {
            this.initCanvasPoseViewer(viewerElement, result);
            return;
        }

        if (!result.receptor_pdb || !result.ligand_pdb) {
            viewerElement.innerHTML = `
                <div class="viewer-placeholder">
                    <p class="placeholder-title">3D payload missing</p>
                    <p class="placeholder-desc">Run prediction again to rebuild the receptor-ligand scene.</p>
                </div>
            `;
            this.updateViewerControlState();
            return;
        }

        const stage = document.createElement('div');
        stage.className = 'viewer-stage';
        viewerElement.appendChild(stage);
        this.addViewerResidueBadge(viewerElement, this.currentResidues.length);

        try {
            this.viewer = $3Dmol.createViewer(stage, { backgroundColor: '#101820' });
            this.viewer.addModel(result.receptor_pdb, 'pdb');
            this.viewer.setStyle({ model: 0 }, { cartoon: { color: 'spectrum', opacity: 0.65 } });

            this.viewer.addModel(result.ligand_pdb, 'pdb');
            this.applyLigandStyle(this.currentLigandStyle);
            this.highlightBindingResidues(this.currentResidues);
            this.addBindingResidueLabels(this.currentResidues);
            this.updateViewerControlState();

            this.viewer.zoomTo({ model: 1 });
            this.viewer.zoom(0.9, 600);
            this.viewer.render();

            setTimeout(() => {
                if (this.viewer?.resize) {
                    this.viewer.resize();
                }
                this.viewer?.render();
            }, 150);
        } catch (error) {
            console.warn('3Dmol viewer failed, using canvas fallback:', error);
            this.viewer = null;
            this.initCanvasPoseViewer(viewerElement, result);
        }
    }

    initCanvasPoseViewer(viewerElement, result) {
        const ligand = this.parsePdbBlock(result.ligand_pdb || '');
        const residues = this.currentResidues.length
            ? this.currentResidues
            : this.getVisualResidues(result.binding_residues || []);

        viewerElement.innerHTML = `
                <div class="canvas-pose-stage">
                    <canvas class="pose-canvas"></canvas>
                    <div class="pose-canvas-hud">
                    <span>${residues.length} Binding Residues Visualized</span>
                    <span>Drag Rotate / Wheel Zoom</span>
                </div>
            </div>
        `;
        this.addViewerResidueBadge(viewerElement, residues.length);

        const canvas = viewerElement.querySelector('.pose-canvas');
        const ctx = canvas.getContext('2d');
        const state = {
            rotX: -0.5,
            rotY: 0.7,
            zoom: 1,
            showLabels: true,
            style: this.currentLigandStyle || 'stick',
            focus: null,
            dragging: false,
            lastX: 0,
            lastY: 0,
        };
        this.labelsVisible = state.showLabels;
        this.surfaceVisible = false;
        this.updateViewerControlState();

        const points = ligand.atoms.concat(residues.map(residue => ({
            x: residue.center?.[0] || 0,
            y: residue.center?.[1] || 0,
            z: residue.center?.[2] || 0,
        })));
        const center = this.computePointCenter(points.length ? points : ligand.atoms);
        const span = Math.max(8, this.computePointSpan(points.length ? points : ligand.atoms, center));

        const render = () => {
            const rect = canvas.getBoundingClientRect();
            const ratio = window.devicePixelRatio || 1;
            canvas.width = Math.max(1, Math.floor(rect.width * ratio));
            canvas.height = Math.max(1, Math.floor(rect.height * ratio));
            ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
            ctx.clearRect(0, 0, rect.width, rect.height);

            const projectedAtoms = ligand.atoms.map(atom => ({
                ...atom,
                screen: this.projectPoint(atom, center, span, state, rect),
            }));
            const projectedResidues = residues.map(residue => ({
                ...residue,
                screen: this.projectPoint({
                    x: residue.center?.[0] || 0,
                    y: residue.center?.[1] || 0,
                    z: residue.center?.[2] || 0,
                }, center, span, state, rect),
            }));

            const gradient = ctx.createLinearGradient(0, 0, rect.width, rect.height);
            gradient.addColorStop(0, '#101820');
            gradient.addColorStop(1, '#172A3A');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, rect.width, rect.height);

            const styleConfig = this.getLigandStyleConfig(state.style).canvas;
            ctx.lineCap = 'round';
            if (styleConfig.showBonds) {
                ligand.bonds.forEach(([from, to]) => {
                    const a = projectedAtoms[from];
                    const b = projectedAtoms[to];
                    if (!a || !b) return;
                    ctx.strokeStyle = styleConfig.bondColor;
                    ctx.lineWidth = styleConfig.bondWidth;
                    ctx.beginPath();
                    ctx.moveTo(a.screen.x, a.screen.y);
                    ctx.lineTo(b.screen.x, b.screen.y);
                    ctx.stroke();
                });
            }

            projectedResidues
                .sort((a, b) => a.screen.z - b.screen.z)
                .forEach((residue, index) => {
                    const focused = state.focus === residue.label;
                    const isClose = index < 6;
                    ctx.fillStyle = focused ? '#F8D66D' : isClose ? '#E7A548' : '#6B7A86';
                    ctx.strokeStyle = focused ? '#FFFFFF' : 'rgba(255, 255, 255, 0.55)';
                    ctx.lineWidth = focused ? 3 : 1;
                    ctx.beginPath();
                    ctx.arc(residue.screen.x, residue.screen.y, focused ? 8 : isClose ? 5 : 3.5, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();

                    if (state.showLabels) {
                        ctx.fillStyle = isClose || focused ? '#FFFFFF' : 'rgba(255, 255, 255, 0.72)';
                        ctx.font = `${isClose || focused ? 10 : 8}px Segoe UI, Arial, sans-serif`;
                        ctx.fillText(`${residue.label} ${Number(residue.min_distance).toFixed(1)}A`, residue.screen.x + 8, residue.screen.y - 8);
                    }
                });

            if (styleConfig.showAtoms) {
                projectedAtoms
                    .sort((a, b) => a.screen.z - b.screen.z)
                    .forEach(atom => {
                        const radius = Math.max(styleConfig.minAtomRadius, styleConfig.atomRadius + atom.screen.z * 0.014);
                        ctx.fillStyle = this.atomColor(atom.element);
                        ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
                        ctx.lineWidth = styleConfig.atomStrokeWidth;
                        ctx.beginPath();
                        ctx.arc(atom.screen.x, atom.screen.y, radius, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.stroke();
                    });
            }

            ctx.fillStyle = 'rgba(255, 255, 255, 0.78)';
            ctx.font = '12px Segoe UI, Arial, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(`${ligand.atoms.length} ligand atoms / ${residues.length} contact residues`, rect.width - 14, rect.height - 16);
            ctx.textAlign = 'left';
        };

        const reset = () => {
            state.rotX = -0.5;
            state.rotY = 0.7;
            state.zoom = 1;
            state.focus = null;
            render();
        };

        canvas.addEventListener('pointerdown', event => {
            state.dragging = true;
            state.lastX = event.clientX;
            state.lastY = event.clientY;
            canvas.setPointerCapture(event.pointerId);
        });

        canvas.addEventListener('pointermove', event => {
            if (!state.dragging) return;
            const dx = event.clientX - state.lastX;
            const dy = event.clientY - state.lastY;
            state.lastX = event.clientX;
            state.lastY = event.clientY;
            state.rotY += dx * 0.01;
            state.rotX += dy * 0.01;
            render();
        });

        canvas.addEventListener('pointerup', () => {
            state.dragging = false;
        });

        canvas.addEventListener('wheel', event => {
            event.preventDefault();
            state.zoom = Math.min(3, Math.max(0.45, state.zoom * (event.deltaY > 0 ? 0.9 : 1.1)));
            render();
        }, { passive: false });

        window.addEventListener('resize', render);
        this.fallbackViewer = {
            render,
            reset,
            toggleLabels: () => {
                state.showLabels = !state.showLabels;
                render();
                return state.showLabels;
            },
            setStyle: style => {
                state.style = style;
                render();
            },
            focusOnLigand: () => {
                state.focus = null;
                state.zoom = 1.25;
                render();
            },
            focusResidue: residue => {
                state.focus = residue?.label || null;
                render();
            },
        };

        this.updateViewerControlState();
        render();
    }

    parsePdbBlock(pdbBlock) {
        const atoms = [];
        const bonds = [];

        pdbBlock.split('\n').forEach(line => {
            const record = line.slice(0, 6).trim();
            if (record === 'ATOM' || record === 'HETATM') {
                const serial = Number.parseInt(line.slice(6, 11), 10);
                const atomName = line.slice(12, 16).trim();
                atoms.push({
                    serial,
                    name: atomName,
                    element: (line.slice(76, 78).trim() || atomName.replace(/[0-9]/g, '').slice(0, 2) || 'C').trim(),
                    x: Number.parseFloat(line.slice(30, 38)),
                    y: Number.parseFloat(line.slice(38, 46)),
                    z: Number.parseFloat(line.slice(46, 54)),
                });
            } else if (record === 'CONECT') {
                const serials = line.slice(6).trim().split(/\s+/).map(value => Number.parseInt(value, 10));
                const from = serials[0];
                serials.slice(1).forEach(to => {
                    const fromIndex = atoms.findIndex(atom => atom.serial === from);
                    const toIndex = atoms.findIndex(atom => atom.serial === to);
                    if (fromIndex >= 0 && toIndex >= 0 && fromIndex < toIndex) {
                        bonds.push([fromIndex, toIndex]);
                    }
                });
            }
        });

        if (!bonds.length) {
            for (let i = 0; i < atoms.length; i += 1) {
                for (let j = i + 1; j < atoms.length; j += 1) {
                    const dx = atoms[i].x - atoms[j].x;
                    const dy = atoms[i].y - atoms[j].y;
                    const dz = atoms[i].z - atoms[j].z;
                    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
                    if (distance > 0.7 && distance < 1.9) {
                        bonds.push([i, j]);
                    }
                }
            }
        }

        return { atoms, bonds };
    }

    computePointCenter(points) {
        if (!points.length) return { x: 0, y: 0, z: 0 };
        return points.reduce((sum, point) => ({
            x: sum.x + point.x / points.length,
            y: sum.y + point.y / points.length,
            z: sum.z + point.z / points.length,
        }), { x: 0, y: 0, z: 0 });
    }

    computePointSpan(points, center) {
        return points.reduce((maxDistance, point) => {
            const dx = point.x - center.x;
            const dy = point.y - center.y;
            const dz = point.z - center.z;
            return Math.max(maxDistance, Math.sqrt(dx * dx + dy * dy + dz * dz));
        }, 1);
    }

    projectPoint(point, center, span, state, rect) {
        const px = point.x - center.x;
        const py = point.y - center.y;
        const pz = point.z - center.z;
        const cosY = Math.cos(state.rotY);
        const sinY = Math.sin(state.rotY);
        const cosX = Math.cos(state.rotX);
        const sinX = Math.sin(state.rotX);
        const x1 = px * cosY - pz * sinY;
        const z1 = px * sinY + pz * cosY;
        const y1 = py * cosX - z1 * sinX;
        const z2 = py * sinX + z1 * cosX;
        const scale = Math.min(rect.width, rect.height) / (span * 2.35) * state.zoom;

        return {
            x: rect.width / 2 + x1 * scale,
            y: rect.height / 2 - y1 * scale,
            z: z2 * scale,
        };
    }

    atomColor(element) {
        const key = String(element || 'C').trim().toUpperCase();
        const colors = {
            C: '#FF37D4',
            N: '#22D3EE',
            O: '#F87171',
            S: '#FDE047',
            P: '#F97316',
            F: '#67E8F9',
            CL: '#34D399',
            BR: '#C084FC',
            I: '#A78BFA',
            H: '#E5E7EB',
        };
        return colors[key] || '#D1D5DB';
    }

    getLigandStyleConfig(style) {
        const configs = {
            stick: {
                viewer: {
                    stick: { colorscheme: 'magentaCarbon', radius: 0.34 },
                    sphere: { colorscheme: 'magentaCarbon', scale: 0.22 }
                },
                canvas: {
                    showBonds: true,
                    showAtoms: true,
                    bondWidth: 5,
                    bondColor: 'rgba(255, 55, 212, 0.82)',
                    atomRadius: 5.8,
                    minAtomRadius: 4.5,
                    atomStrokeWidth: 1.2
                }
            },
            sphere: {
                viewer: { sphere: { colorscheme: 'magentaCarbon', scale: 0.36 } },
                canvas: {
                    showBonds: false,
                    showAtoms: true,
                    bondWidth: 0,
                    bondColor: 'rgba(255, 55, 212, 0.82)',
                    atomRadius: 12.5,
                    minAtomRadius: 9,
                    atomStrokeWidth: 1.5
                }
            },
            line: {
                viewer: { line: { colorscheme: 'magentaCarbon', linewidth: 6 } },
                canvas: {
                    showBonds: true,
                    showAtoms: true,
                    bondWidth: 2,
                    bondColor: 'rgba(255, 55, 212, 0.9)',
                    atomRadius: 3.4,
                    minAtomRadius: 2.8,
                    atomStrokeWidth: 1
                }
            },
            ballstick: {
                viewer: {
                    stick: { colorscheme: 'magentaCarbon', radius: 0.28 },
                    sphere: { colorscheme: 'magentaCarbon', scale: 0.28 }
                },
                canvas: {
                    showBonds: true,
                    showAtoms: true,
                    bondWidth: 3,
                    bondColor: 'rgba(230, 237, 243, 0.72)',
                    atomRadius: 8.8,
                    minAtomRadius: 5.8,
                    atomStrokeWidth: 1.4
                }
            }
        };

        return configs[style] || configs.stick;
    }

    applyLigandStyle(style) {
        this.currentLigandStyle = style || 'stick';
        if (!this.viewer) return;

        this.viewer.setStyle({ model: 1 }, this.getLigandStyleConfig(this.currentLigandStyle).viewer);
    }

    highlightBindingResidues(residues) {
        if (!this.viewer) return;

        residues.forEach((residue, index) => {
            const close = index < 6;
            this.viewer.addStyle(
                { model: 0, chain: residue.chain === '-' ? ' ' : residue.chain, resi: residue.res_id },
                close
                    ? { stick: { color: '#E7A548', radius: 0.11 } }
                    : { line: { color: '#7A8791', linewidth: 1 } }
            );
        });
    }

    addBindingResidueLabels(residues) {
        if (!this.viewer) return;

        this.clearResidueLabels();
        this.residueLabels = residues.map((residue, index) => {
            const [x, y, z] = residue.center || [0, 0, 0];
            const close = index < 6;
            return this.viewer.addLabel(residue.label, {
                position: { x, y, z },
                backgroundColor: close ? '#14324A' : '#2D3A40',
                backgroundOpacity: close ? 0.72 : 0.48,
                fontColor: '#FFFFFF',
                fontSize: close ? 10 : 8,
                borderThickness: 1,
                borderColor: close ? '#7DD3FC' : '#7A8791'
            });
        });
        this.labelsVisible = true;
    }

    clearResidueLabels() {
        if (!this.viewer || !this.residueLabels.length) return;
        this.residueLabels.forEach(label => this.viewer.removeLabel(label));
        this.residueLabels = [];
    }

    getBindingPocketSelection() {
        const residueSelections = (this.currentResidues || [])
            .filter(residue => residue.res_id !== undefined && residue.res_id !== null)
            .map(residue => ({
                model: 0,
                chain: residue.chain === '-' ? ' ' : residue.chain,
                resi: residue.res_id
            }));

        return residueSelections.length ? { or: residueSelections } : { model: 0 };
    }

    updateViewerControlState() {
        const setPressed = (id, pressed) => {
            const button = document.getElementById(id);
            if (!button) return;
            button.classList.toggle('active', Boolean(pressed));
            button.setAttribute('aria-pressed', pressed ? 'true' : 'false');
        };

        setPressed('toggleLabelsBtn', this.labelsVisible);
        setPressed('toggleSurfaceBtn', this.surfaceVisible);

        const surfaceButton = document.getElementById('toggleSurfaceBtn');
        if (surfaceButton) {
            const surfaceAvailable = Boolean(this.viewer) && !this.fallbackViewer;
            surfaceButton.disabled = !surfaceAvailable;
            surfaceButton.title = surfaceAvailable
                ? 'Toggle binding-pocket surface'
                : 'Surface requires the full 3D viewer';
        }

        const ligandFocusButton = document.getElementById('ligandFocusBtn');
        if (ligandFocusButton) {
            ligandFocusButton.disabled = !this.viewer && !this.fallbackViewer;
        }

        const fullViewerButtonTitles = {
            bindingPocketBtn: 'Show binding pocket',
            interactionMapBtn: 'Show interaction map'
        };
        Object.keys(fullViewerButtonTitles).forEach(id => {
            const button = document.getElementById(id);
            if (!button) return;
            const fullViewerAvailable = Boolean(this.viewer) && !this.fallbackViewer;
            button.disabled = !fullViewerAvailable;
            button.title = fullViewerAvailable
                ? fullViewerButtonTitles[id]
                : 'Requires the full 3D viewer';
        });

        const styleSelect = document.getElementById('styleSelect');
        if (styleSelect) {
            styleSelect.disabled = !this.viewer && !this.fallbackViewer;
            styleSelect.value = this.currentLigandStyle || 'stick';
        }
    }

    resetView() {
        if (this.fallbackViewer) {
            this.fallbackViewer.reset();
            this.updateViewerControlState();
            return;
        }

        if (this.viewer) {
            this.restoreDefaultMolecularScene();
            this.viewer.zoomTo({ model: 1 });
            this.viewer.render();
            this.updateViewerControlState();
        }
    }

    toggleLabels() {
        if (this.fallbackViewer) {
            this.labelsVisible = this.fallbackViewer.toggleLabels();
            this.updateViewerControlState();
            return;
        }

        if (this.viewer) {
            if (this.labelsVisible) {
                this.clearResidueLabels();
                this.labelsVisible = false;
            } else {
                this.addBindingResidueLabels(this.currentResidues || []);
                this.labelsVisible = true;
            }
            this.viewer.render();
            this.updateViewerControlState();
        }
    }

    toggleSurface() {
        if (this.fallbackViewer) {
            this.showAlert('Surface view requires the full 3Dmol viewer', 'info');
            return;
        }

        if (this.viewer) {
            if (this.surfaceVisible && this.surfaceId !== null) {
                const surfaceToRemove = this.surfaceId;
                this.surfaceId = null;
                this.surfaceVisible = false;
                Promise.resolve(surfaceToRemove).then(surfaceId => {
                    if (surfaceId !== undefined && surfaceId !== null) {
                        this.viewer?.removeSurface(surfaceId);
                        this.viewer?.render();
                    }
                });
            } else {
                const surface = this.viewer.addSurface(
                    $3Dmol.SurfaceType.VDW,
                    { opacity: 0.18, color: '#4FB8A0' },
                    this.getBindingPocketSelection()
                );
                this.surfaceId = surface;
                this.surfaceVisible = true;
                Promise.resolve(surface).then(surfaceId => {
                    if (this.surfaceVisible) {
                        this.surfaceId = surfaceId;
                        this.viewer?.render();
                    }
                }).catch(error => {
                    console.warn('Surface generation failed:', error);
                    this.surfaceId = null;
                    this.surfaceVisible = false;
                    this.updateViewerControlState();
                });
            }
            this.viewer.render();
            this.updateViewerControlState();
        }
    }

    restoreDefaultMolecularScene() {
        if (!this.viewer) return;

        const keepLabels = this.labelsVisible;
        this.viewer.setStyle({}, {});
        this.viewer.setStyle({ model: 0 }, { cartoon: { color: 'spectrum', opacity: 0.65 } });
        this.applyLigandStyle(this.currentLigandStyle);
        this.highlightBindingResidues(this.currentResidues || []);

        if (keepLabels) {
            this.addBindingResidueLabels(this.currentResidues || []);
        } else {
            this.clearResidueLabels();
            this.labelsVisible = false;
        }
    }

    focusOnLigand() {
        if (this.fallbackViewer) {
            this.fallbackViewer.focusOnLigand();
            this.showAlert('Focused on ligand', 'success');
            return;
        }

        if (!this.viewer) return;

        this.restoreDefaultMolecularScene();
        this.viewer.zoomTo({ model: 1 });
        this.viewer.render();
        this.showAlert('Focused on ligand', 'success');
    }

    showBindingPocket() {
        if (this.fallbackViewer) {
            this.showAlert('Binding pocket view requires the full 3Dmol viewer', 'info');
            return;
        }

        if (!this.viewer) return;

        this.viewer.setStyle({}, {});
        this.viewer.setStyle({ model: 1 }, {
            stick: { colorscheme: 'magentaCarbon', radius: 0.34 },
            sphere: { colorscheme: 'magentaCarbon', scale: 0.22 }
        });

        const pocketSelection = this.getBindingPocketSelection();
        this.viewer.setStyle({ model: 0 }, { cartoon: { color: '#1B6D82', opacity: 0.28 } });
        this.viewer.addStyle(pocketSelection, { stick: { color: '#E7A548', radius: 0.15 } });

        if (this.labelsVisible) {
            this.addBindingResidueLabels(this.currentResidues || []);
        }

        this.viewer.zoomTo({ model: 1 });
        this.viewer.zoom(0.82, 450);
        this.viewer.render();
        this.showAlert('Showing binding pocket', 'success');
    }

    showInteractionMap() {
        if (this.fallbackViewer) {
            this.showAlert('Interaction map requires the full 3Dmol viewer', 'info');
            return;
        }

        if (!this.viewer || !this.currentResidues || this.currentResidues.length === 0) {
            this.showAlert('No interaction data available', 'warning');
            return;
        }

        this.viewer.setStyle({}, {});
        this.viewer.setStyle({ model: 0 }, { cartoon: { color: '#1B6D82', opacity: 0.22 } });
        this.viewer.setStyle({ model: 1 }, {
            stick: { colorscheme: 'magentaCarbon', radius: 0.36 },
            sphere: { colorscheme: 'magentaCarbon', scale: 0.24 }
        });

        this.currentResidues.forEach(residue => {
            const interactions = residue.interactions || [];
            let color = '#7A8791';
            
            if (interactions.includes('H-Bond')) color = '#4CAF50';
            else if (interactions.includes('Hydrophobic')) color = '#FF9800';
            else if (interactions.includes('Pi-Stacking')) color = '#9C27B0';
            else if (interactions.includes('Salt Bridge')) color = '#F44336';
            
            this.viewer.setStyle(
                { model: 0, chain: residue.chain === '-' ? ' ' : residue.chain, resi: residue.res_id },
                { stick: { color, radius: 0.18 } }
            );
        });

        if (this.labelsVisible) {
            this.addBindingResidueLabels(this.currentResidues || []);
        }

        this.viewer.zoomTo({ model: 1 });
        this.viewer.zoom(0.86, 450);
        this.viewer.render();
        this.showAlert('Showing interaction map', 'success');
    }

    toggleSurfaceView() {
        this.toggleSurface();
    }

    changeStyle(style) {
        this.currentLigandStyle = style || 'stick';

        if (this.fallbackViewer) {
            this.fallbackViewer.setStyle(this.currentLigandStyle);
            this.updateViewerControlState();
            return;
        }

        if (!this.viewer) return;

        this.applyLigandStyle(this.currentLigandStyle);
        this.highlightBindingResidues(this.currentResidues || []);
        this.viewer.render();
        this.updateViewerControlState();
    }

    focusResidue(residue) {
        if (this.fallbackViewer) {
            this.fallbackViewer.focusResidue(residue);
            return;
        }

        if (!this.viewer || !residue) return;

        this.viewer.zoomTo({
            model: 0,
            chain: residue.chain === '-' ? ' ' : residue.chain,
            resi: residue.res_id
        });
        this.viewer.render();
    }

    displayDrugLikeness(drugLikeness) {
        if (!drugLikeness) return;

        // Update assessment badge
        const badge = document.getElementById('druglikenessAssessment');
        if (badge) {
            badge.className = `druglikeness-badge ${drugLikeness.assessment_class}`;
            badge.querySelector('.badge-text').textContent = drugLikeness.assessment;
        }

        // Update metrics cards
        const metrics = drugLikeness.metrics || {};
        Object.keys(metrics).forEach(key => {
            const metric = metrics[key];
            const card = document.getElementById(`metric-${key}`);
            if (card) {
                const statusEl = card.querySelector('.metric-status');
                const detailsEl = card.querySelector('.metric-details');
                
                if (statusEl) {
                    statusEl.textContent = metric.passed ? 'Pass' : 'Review';
                    statusEl.className = `metric-status ${metric.passed ? 'pass' : 'fail'}`;
                }
                
                if (detailsEl) {
                    detailsEl.textContent = metric.details;
                }
            }
        });

        // Render radar chart
        this.renderDrugLikenessRadar(drugLikeness.radar_data);
    }

    renderDrugLikenessRadar(radarData) {
        const canvas = document.getElementById('druglikenessRadar');
        if (!canvas || !radarData) return;

        const ctx = canvas.getContext('2d');
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = Math.min(centerX, centerY) - 60;
        const labels = radarData.labels || [];
        const values = radarData.values || [];
        const numPoints = labels.length;
        if (!numPoints || !values.length) return;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw background circles
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;
        for (let i = 1; i <= 5; i++) {
            ctx.beginPath();
            ctx.arc(centerX, centerY, (radius / 5) * i, 0, 2 * Math.PI);
            ctx.stroke();
        }

        // Draw axes
        ctx.strokeStyle = '#d0d0d0';
        ctx.lineWidth = 1;
        for (let i = 0; i < numPoints; i++) {
            const angle = (Math.PI * 2 * i) / numPoints - Math.PI / 2;
            const x = centerX + radius * Math.cos(angle);
            const y = centerY + radius * Math.sin(angle);
            
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(x, y);
            ctx.stroke();
        }

        // Draw data polygon
        ctx.beginPath();
        ctx.fillStyle = 'rgba(55, 138, 221, 0.2)';
        ctx.strokeStyle = 'rgba(55, 138, 221, 0.8)';
        ctx.lineWidth = 2;

        for (let i = 0; i < numPoints; i++) {
            const angle = (Math.PI * 2 * i) / numPoints - Math.PI / 2;
            const value = values[i] / 100; // Normalize to 0-1
            const x = centerX + radius * value * Math.cos(angle);
            const y = centerY + radius * value * Math.sin(angle);
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // Draw data points
        ctx.fillStyle = '#378ADD';
        for (let i = 0; i < numPoints; i++) {
            const angle = (Math.PI * 2 * i) / numPoints - Math.PI / 2;
            const value = values[i] / 100;
            const x = centerX + radius * value * Math.cos(angle);
            const y = centerY + radius * value * Math.sin(angle);
            
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fill();
        }

        // Draw labels
        ctx.fillStyle = '#2C2C2A';
        ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        for (let i = 0; i < numPoints; i++) {
            const angle = (Math.PI * 2 * i) / numPoints - Math.PI / 2;
            const labelRadius = radius + 35;
            const x = centerX + labelRadius * Math.cos(angle);
            const y = centerY + labelRadius * Math.sin(angle);
            
            // Adjust text alignment based on position
            if (Math.abs(x - centerX) < 5) {
                ctx.textAlign = 'center';
            } else if (x > centerX) {
                ctx.textAlign = 'left';
            } else {
                ctx.textAlign = 'right';
            }
            
            ctx.fillText(labels[i], x, y);
            
            // Draw percentage value
            ctx.font = '9px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
            ctx.fillStyle = '#888780';
            ctx.fillText(`${Math.round(values[i])}%`, x, y + 12);
            ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
            ctx.fillStyle = '#2C2C2A';
        }

        // Draw center score
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        ctx.fillStyle = '#378ADD';
        const avgScore = values.reduce((a, b) => a + b, 0) / values.length;
        ctx.fillText(`${Math.round(avgScore)}%`, centerX, centerY);
        ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        ctx.fillStyle = '#888780';
        ctx.fillText('Overall', centerX, centerY + 15);
    }

    setText(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    updateProgress(percent, message) {
        const progressBar = document.getElementById('progressBar');
        const progressPercent = document.getElementById('progressPercent');
        const progressText = document.getElementById('progressText');
        
        progressBar.style.width = `${percent}%`;
        progressPercent.textContent = `${percent}%`;
        progressText.textContent = message;

        // Update step indicators
        const steps = ['step1', 'step2', 'step3', 'step4'];
        steps.forEach((stepId, index) => {
            const step = document.getElementById(stepId);
            const threshold = (index + 1) * 25;
            
            if (percent >= threshold) {
                step.classList.add('completed');
                step.classList.remove('active');
            } else if (percent >= threshold - 25 && percent < threshold) {
                step.classList.add('active');
                step.classList.remove('completed');
            } else {
                step.classList.remove('active', 'completed');
            }
        });
    }

    async downloadReport(format) {
        if (!this.currentResult) {
            this.showAlert('No results available to download', 'error');
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/download/${format}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.currentResult)
            });

            if (!response.ok) {
                throw new Error('Download failed');
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const filenames = {
                pdf: 'biovanguard_report.pdf',
                json: 'biovanguard_result.json'
            };
            a.download = filenames[format] || `biovanguard_export.${format}`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            const label = format === 'json' ? 'JSON data' : 'PDF report';
            this.showAlert(`${label} downloaded successfully`, 'success');
        } catch (error) {
            console.error('Download error:', error);
            this.showAlert(`Download failed: ${error.message}`, 'error');
        }
    }

    showAlert(message, type = 'info') {
        const container = document.getElementById('alertContainer');
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        
        alert.innerHTML = `<span>${message}</span>`;
        
        container.appendChild(alert);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            alert.style.opacity = '0';
            alert.style.transform = 'translateX(400px)';
            setTimeout(() => alert.remove(), 300);
        }, 5000);
    }

    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new BioVanguardApp();
});

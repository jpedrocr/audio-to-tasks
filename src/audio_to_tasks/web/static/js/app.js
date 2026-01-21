// AudioToTasks Web UI
document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('audio-file');
    const processBtn = document.getElementById('process-btn');
    const uploadForm = document.getElementById('upload-form');
    const resultsSection = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const loadingText = document.getElementById('loading-text');

    let selectedFile = null;

    // Drop zone handlers
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        dropZone.textContent = '';
        const p1 = document.createElement('p');
        const strong = document.createElement('strong');
        strong.textContent = 'Selected: ';
        p1.appendChild(strong);
        p1.appendChild(document.createTextNode(file.name));
        dropZone.appendChild(p1);

        const p2 = document.createElement('p');
        p2.className = 'meta';
        p2.textContent = formatFileSize(file.size);
        dropZone.appendChild(p2);

        processBtn.disabled = false;
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    // Form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        const language = document.getElementById('language').value;

        // Show loading
        loadingDiv.hidden = false;
        resultsSection.hidden = true;
        loadingText.textContent = 'Transcribing audio...';

        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        if (language) {
            formData.append('language', language);
        }

        try {
            loadingText.textContent = 'Transcribing and extracting tasks...';

            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            if (result.success) {
                displayResults(result);
            } else {
                alert('Error: ' + (result.error || 'Unknown error'));
            }
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            loadingDiv.hidden = true;
        }
    });

    function displayResults(result) {
        // Show transcription
        document.getElementById('transcription-text').textContent =
            result.transcription.text;
        document.getElementById('transcription-meta').textContent =
            `Language: ${result.transcription.language} | ` +
            `Duration: ${formatDuration(result.transcription.duration_seconds)} | ` +
            `Processing time: ${result.processing_time_seconds.toFixed(1)}s`;

        // Show tasks
        const taskList = document.getElementById('task-list');
        taskList.textContent = '';

        if (result.tasks.tasks.length === 0) {
            const li = document.createElement('li');
            li.className = 'no-tasks';
            li.textContent = 'No tasks found in this recording';
            taskList.appendChild(li);
        } else {
            result.tasks.tasks.forEach((task, index) => {
                const li = document.createElement('li');
                li.className = `task priority-${task.priority}`;

                // Task number
                const taskNumber = document.createElement('span');
                taskNumber.className = 'task-number';
                taskNumber.textContent = index + 1;
                li.appendChild(taskNumber);

                // Task content
                const taskContent = document.createElement('div');
                taskContent.className = 'task-content';

                const title = document.createElement('strong');
                title.textContent = task.title;
                taskContent.appendChild(title);

                if (task.description) {
                    const desc = document.createElement('p');
                    desc.textContent = task.description;
                    taskContent.appendChild(desc);
                }

                const priorityBadge = document.createElement('span');
                priorityBadge.className = 'priority-badge';
                priorityBadge.textContent = task.priority;
                taskContent.appendChild(priorityBadge);

                if (task.assignee) {
                    const assignee = document.createElement('span');
                    assignee.className = 'assignee';
                    assignee.textContent = '@' + task.assignee;
                    taskContent.appendChild(assignee);
                }

                if (task.tags && task.tags.length > 0) {
                    const tags = document.createElement('span');
                    tags.className = 'assignee';
                    tags.textContent = task.tags.map(t => '#' + t).join(' ');
                    taskContent.appendChild(tags);
                }

                li.appendChild(taskContent);
                taskList.appendChild(li);
            });
        }

        resultsSection.hidden = false;
    }

    function formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
});

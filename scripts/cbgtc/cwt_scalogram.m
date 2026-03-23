function cwt_scalogram(desired_dataset)

% ----- Instruction ------
% matlab -nodisplay -nosplash -nodesktop -r "cwt_scalogram('23Jan2026'); exit"
% ------------------------

% -------------------------
% Configuration
% -------------------------
channels = 0:3;
regions = {'cortex', 'thalamus', ''};

cortex_nuclei   = {'CSN', 'PTN', 'IN'};
thalamus_nuclei = {'MD', 'TRN'};
other_nuclei    = {'MSN', 'STN', 'FSI', 'GPi', 'GPe'};

outdir = fullfile(desired_dataset, 'figures');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

decayfolderid = containers.Map( ...
    {'0','1','2','3','4','5','6','7','8','9', ...
     '10','11','12','13','14','15','16','17','18','19'}, ...
    [ 0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, ...
      0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00 ] );

% -------------------------
% Find all model_9_sim*
% -------------------------
sim_dirs = dir(fullfile(desired_dataset, 'data_recorded', 'model_9*'));

for s = 1:numel(sim_dirs)
    sim_path = fullfile(sim_dirs(s).folder, sim_dirs(s).name);

    fprintf('Processing %s\n', sim_path);

    % -------- Determine region from folder name --------
    base_path = sim_path;   % <-- THIS IS THE KEY LINE

    if contains(sim_dirs(s).name, 'cortex')
        region = 'cortex';
        nuclei = cortex_nuclei;

    elseif contains(sim_dirs(s).name, 'thalamus')
        region = 'thalamus';
        nuclei = thalamus_nuclei;

    else
        region = '';
        nuclei = other_nuclei;
    end


    % -------- Nucleus loop --------
    for n = 1:numel(nuclei)
        nucleus = nuclei{n};

        spr_all = [];
        t_ms = [];

        for ch = channels
            fname = sprintf('smoothed_pop_rate_%s_%d.csv', nucleus, ch);
            fpath = fullfile(base_path, fname);

            if ~exist(fpath, 'file')
                warning('Missing %s', fpath);
                continue
            end

            T = readtable(fpath);

            if isempty(t_ms)
                t_ms = T.t_ms;
            end

            spr_all(:, end+1) = T.smoothed_pop_rate; %#ok<SAGROW>
        end

        if size(spr_all, 2) < 1
            continue
        end

        % -------- Average across channels --------
        spr_mean = mean(spr_all, 2);

        % -------- Sampling rate --------
        dt = (t_ms(end) - t_ms(end-1)) * 1e-3; % seconds
        fs = 1 / dt;

        % -------- CWT --------
        [wt, f] = cwt(spr_mean, 'bump', fs);

        % -------- Extract decay ---------
        tokens = regexp(sim_dirs(s).name, 'sim(\d+)', 'tokens');

        if isempty(tokens)
            warning('Could not extract sim id from %s', sim_dirs(s).name);
            sim_id = NaN;
        else
            sim_id = str2double(tokens{1}{1});
        end

        % Look up decay percentage
        key = num2str(sim_id);

        if isKey(decayfolderid, key)
            decay = decayfolderid(key) * 100;   % percentage
        else
            warning('sim id %s not in decay map', key);
            decay = NaN;
        end

        % -------- Plot (non-GUI) --------
        fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 600]);

        tiledlayout(2,1, 'Padding', 'compact', 'TileSpacing', 'compact');

        % Population rate
        nexttile;
        plot(t_ms * 1e-3, spr_mean, 'k', 'LineWidth', 1);
        xlabel('Time (s)');
        ylabel('Population rate');
        % title(sprintf('%s | %s', sim_dirs(s).name, nucleus), 'Interpreter', 'none');
        title(sprintf('%s | %s | decay = %.0f%%', ...
                      sim_dirs(s).name, nucleus, decay), ...
                      'Interpreter', 'none');

        % CWT
        nexttile;
        % imagesc(t_ms * 1e-3, f, abs(wt));
        % ----- y-axis to match Magnitude Scalogram ------
        imagesc(t_ms * 1e-3, log2(f), abs(wt).^2); % MATLAB uses magnitude-squared
        yticks = 2.^(floor(log2(min(f))):ceil(log2(max(f))));
        set(gca, ...
            'YTick', log2(yticks), ...
            'YTickLabel', yticks);

        axis xy;
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        title('CWT (bump)');
        colormap turbo;
        colorbar;

        % -------- Save uncompressed --------
        outname = sprintf('%s_%s.png', sim_dirs(s).name, nucleus);
        savepath = fullfile(outdir, outname);
        print(fig, savepath, '-dpng', '-r300');

        close(fig);

        % ------- 2nd Plot --------
        fig2 = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 600]);

        cwt(spr_mean, 'bump', fs);
        title(sprintf('Magnitude Scalogram of %s | decay = %.0f%%', ...
                      nucleus, decay), ...
                      'Interpreter', 'none');

        % -------- Save uncompressed --------
        outname = sprintf('%s_%s_scalogram.png', sim_dirs(s).name, nucleus);
        savepath = fullfile(outdir, outname);
        print(fig2, savepath, '-dpng', '-r300');

        close(fig2);
    end
end

fprintf('Done.\n');
end

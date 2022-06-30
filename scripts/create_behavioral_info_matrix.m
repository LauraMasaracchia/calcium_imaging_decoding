%% ANIMAL 232 MESSSED UP INFO - CREATING NEW ONES

%% retrieve right behavioral data - non airpuff

trial_start = SessionData.TrialStartTimestamp;
trial_end= SessionData.TrialEndTimestamp;

ntrials = SessionData.nTrials;


%% compare with timestamps

%load timestamps

%start_index = [1:5:ntrials*5];
start_index = [1:5:615];

timestamp_start = timestamps(start_index);
trial_start_sync = trial_start + timestamp_start(1) - trial_start(1);
trial_start_sync(1:123) - timestamp_start  

%%
%end_index = [5:5:ntrials*5];
end_index = [5:5:615];
timestamp_end = timestamps(end_index);

trial_end_sync = trial_end + timestamps(1) - trial_start(1);

trial_end_sync(1:123) - timestamp_end


%% if it is good enough, just get the rest and save
ntrials=123;

trial_start_new = trial_start_sync(1:ntrials);
trial_end_new = trial_end_sync(1:ntrials);


trial_stimulus_on = zeros(1,ntrials);
trial_response = zeros(1,ntrials);

trial_resp_to_end = zeros(1,ntrials);


trial_is_reward = zeros(1,ntrials);
trial_is_right_lick = zeros(1,ntrials);


for i=1:ntrials
    trial_stimulus_on(i) = trial_start_new(i) + SessionData.RawEvents.Trial{1,i}.States.WaitStimulus(2);
    trial_response(i) = trial_start_new(i) + SessionData.RawEvents.Trial{1,i}.States.DeliverStimulus(2);
    trial_resp_to_end(i) = trial_start_new(i) + SessionData.RawEvents.Trial{1,i}.States.Wait(2) - SessionData.RawEvents.Trial{1,i}.States.DeliverStimulus(2);
    if ~isnan(SessionData.RawEvents.Trial{1,i}.States.Reward(1))
        trial_is_reward(i) = 1;
    end
    if isfield(SessionData.RawEvents.Trial{1,i}.Events, 'Port2In')
        if ismember(SessionData.RawEvents.Trial{1,i}.States.DeliverStimulus(2),SessionData.RawEvents.Trial{1,i}.Events.Port2In)
            trial_is_right_lick(i)=1;
        end
    end
    
end



%% save new trial info matrix
    
save('laura/trial_info_232laura_20210508.mat', 'trial_start_new', 'trial_stimulus_on', 'trial_response', 'trial_is_reward','trial_is_right_lick', 'trial_end_new')

save('laura/fixed_timestamps_232laura_20210508.mat', 'timestamps');


%%
%%
%%
%%
%% WITH THE AIRPUFF THE NAMES ARE DIFFERENT - need a new script


%%

trial_start = SessionData.TrialStartTimestamp;
trial_end= SessionData.TrialEndTimestamp;

ntrials = SessionData.nTrials;


%% compare with timestamps

%load timestamps

%start_index = [1:5:ntrials*5];
start_index = [1:4:611];

timestamp_start = timestamps(start_index);
trial_start_sync = trial_start + timestamp_start(1) - trial_start(1);
trial_start_sync(1:152) - timestamp_start(1:152)  

%%
%end_index = [5:5:ntrials*5];
end_index = [4:4:611];
timestamp_end = timestamps(end_index);

trial_end_sync = trial_end + timestamps(1) - trial_start(1);

trial_end_sync(1:152) - timestamp_end(1:152)


%% if it is good enough, just get the rest and save
ntrials=152;

trial_start_new = trial_start_sync(1:ntrials);
trial_end_new = trial_end_sync(1:ntrials);


trial_stimulus_on = zeros(1,ntrials);
trial_response = zeros(1,ntrials);

trial_resp_to_end = zeros(1,ntrials);
trial_is_reward = zeros(1,ntrials);
trial_is_right_lick = zeros(1,ntrials);


for i=1:ntrials
    trial_stimulus_on(i) = trial_start_new(i) + SessionData.RawEvents.Trial{1,i}.States.WaitStimulus(2);
    trial_response(i) = trial_start_new(i) + SessionData.RawEvents.Trial{1,i}.States.DeliverStimulus(2);
    %trial_resp_to_end(i) = trial_start_new(i) + SessionData.RawEvents.Trial{1,i}.States.Wait(2) - SessionData.RawEvents.Trial{1,i}.States.DeliverStimulus(2);
    if ~isnan(SessionData.RawEvents.Trial{1,i}.States.Reward(1))
        trial_is_reward(i) = 1;
    end
    if isfield(SessionData.RawEvents.Trial{1,i}.Events, 'Port2In')
        if ismember(SessionData.RawEvents.Trial{1,i}.States.DeliverStimulus(2),SessionData.RawEvents.Trial{1,i}.Events.Port2In)
            trial_is_right_lick(i)=1;
        end
    end
    
end


%% save new ones
save('laura/trial_info_232laura_20210519.mat', 'trial_start_new', 'trial_stimulus_on', 'trial_response', 'trial_is_reward','trial_is_right_lick', 'trial_end_new')

save('laura/fixed_timestamps_232laura_20210519.mat', 'timestamps');



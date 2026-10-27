; Inno Setup 6 script: the per-user Windows installer of "Assetto Corsa Map Generator".
;
;   iscc /DAppVersion=0.4.0 /DVersionNumeric=0.4.0.0 packaging\installer.iss
;
; Packs the PyInstaller one-folder build (dist\AssettoCorsaMapGenerator, see acmg.spec) into
; dist\AssettoCorsaMapGenerator-Setup-<AppVersion>.exe. Built and tested by
; .github/workflows/windows-app.yml.
;
; - No admin rights: installs for the current user only, into
;   %LOCALAPPDATA%\Programs\Assetto Corsa Map Generator, with a Start Menu shortcut, an
;   optional desktop shortcut and a normal uninstaller (Settings > Apps).
; - User data never lives in the install folder: settings, the Mapillary token, the offline
;   Ontario roads, the cache and built tracks are in %LOCALAPPDATA%\AssettoCorsaMapGenerator
;   (driftmaps.app.settings.app_data_dir), so installing a new version over an old one keeps
;   them. The uninstaller asks before deleting that folder (a silent uninstall keeps it).
; - Updates: the app's "Install" button runs this setup with
;   /VERYSILENT /SUPPRESSMSGBOXES /CURRENTUSER /NORESTART /relaunch=1 [/port=N]
;   (driftmaps.app.updates). Setup waits for the running app to quit (it holds the mutex
;   AppMutexName, see packaging/launcher.py), installs, then starts the new version with
;   --after-update so the open page reloads itself.

#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif
#ifndef VersionNumeric
  #define VersionNumeric "0.0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\AssettoCorsaMapGenerator"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif

#define AppName "Assetto Corsa Map Generator"
#define AppExe "AssettoCorsaMapGenerator.exe"
#define DataDirName "AssettoCorsaMapGenerator"
#define AppMutexName "AssettoCorsaMapGenerator.Running"
#define RepoUrl "https://github.com/cooldude-7/MapsAssetoCorsa"

[Setup]
AppId={{0E3C3A23-8572-451D-AA3A-65E2387AF237}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=cooldude-7
AppPublisherURL={#RepoUrl}
AppSupportURL={#RepoUrl}/issues
AppUpdatesURL={#RepoUrl}/releases/latest
VersionInfoVersion={#VersionNumeric}
VersionInfoProductName={#AppName}
VersionInfoDescription={#AppName} Setup
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\{#AppName}
DisableDirPage=yes
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UsePreviousAppDir=no
OutputDir={#OutputDir}
OutputBaseFilename=AssettoCorsaMapGenerator-Setup-{#AppVersion}
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
WizardStyle=modern
Compression=lzma2/normal
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
; Files in use: close the app (silently too), but never let Restart Manager restart it: the
; [Run] entries below do that.
CloseApplications=force
RestartApplications=no
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; The previous version's program files (PyInstaller's _internal folder changes between
; versions). Only program files live here; user data is in %LOCALAPPDATA%\{#DataDirName}.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; A normal install: "Launch" tick box on the last page.
Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
; An in-app update (silent, /relaunch=1): start the new version straight away.
Filename: "{app}\{#AppExe}"; Parameters: "--after-update --port {param:port|8765}"; WorkingDir: "{app}"; Flags: nowait; Check: RelaunchAfterUpdate

[Code]
function RelaunchAfterUpdate: Boolean;
begin
  Result := WizardSilent and (ExpandConstant('{param:relaunch|0}') = '1');
end;

function AppRunning: Boolean;
begin
  Result := CheckForMutexes('{#AppMutexName}');
end;

function InitializeSetup: Boolean;
var
  Waited: Integer;
begin
  Result := True;
  if WizardSilent then begin
    { An update: the app closes itself right after starting this setup. Wait up to a minute;
      if it is still open then, CloseApplications=force closes it. }
    Waited := 0;
    while AppRunning and (Waited < 120) do begin
      Sleep(500);
      Waited := Waited + 1;
    end;
    if AppRunning then
      Log('The app is still running after 60 s; Setup will close it.')
    else
      Log(Format('The app is not running (waited %d ms).', [Waited * 500]));
  end else begin
    while AppRunning do begin
      if MsgBox('{#AppName} is open. Close its window (the black one that says "keep this ' +
                'window open"), then click OK.', mbInformation, MB_OKCANCEL) = IDCANCEL then begin
        Result := False;
        Exit;
      end;
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir, CacheDir: String;
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;
  DataDir := ExpandConstant('{localappdata}\{#DataDirName}');
  CacheDir := ExpandConstant('{localappdata}\driftmaps');
  if not DirExists(DataDir) and not DirExists(CacheDir) then
    Exit;
  if UninstallSilent then begin
    Log('Kept the user data in ' + DataDir);
    Exit;
  end;
  if MsgBox('Also delete your settings, downloaded road data, cache and built tracks?' + #13#10#13#10 +
            'They are in:' + #13#10 + DataDir + #13#10#13#10 +
            'Choose No to keep them, for example to install the app again later. Tracks ' +
            'already installed in Assetto Corsa stay either way.',
            mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then begin
    DelTree(DataDir, True, True, True);
    DelTree(CacheDir, True, True, True);
  end else
    MsgBox('Your settings and data were kept in:' + #13#10 + DataDir, mbInformation, MB_OK);
end;

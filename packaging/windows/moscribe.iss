; Inno Setup script — builds MoScribe-Setup-x.y.z.exe
;   iscc packaging\windows\moscribe.iss
#define AppName "MoScribe"
#define AppVersion "1.0.0"
#define AppPublisher "kmail.at"
#define AppURL "https://github.com/mokmail/moscribe"

[Setup]
AppId={{9C4A1F2E-5B7D-4C3A-9E11-MOSCRIBE0001}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputBaseFilename=MoScribe-windows-x64-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\..\moscribe\static\brand\app.ico
UninstallDisplayIcon={app}\MoScribe.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\..\dist\MoScribe\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\MoScribe.exe"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\MoScribe.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\MoScribe.exe"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\MoScribe\jobs"

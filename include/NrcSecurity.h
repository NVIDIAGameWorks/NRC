/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <windows.h>
#include <Softpub.h>

#define GetProc(hModule, procName, proc) (((NULL == proc) && (NULL == (*((FARPROC*)&proc) = GetProcAddress(hModule, procName)))) ? FALSE : TRUE)
typedef LONG(WINAPI* PfnWinVerifyTrust)(IN HWND hwnd, IN GUID* pgActionID, IN LPVOID pWVTData);
static PfnWinVerifyTrust pfnWinVerifyTrust = NULL;

namespace nrc
{
namespace security
{
// Microsoft's approach https://learn.microsoft.com/en-us/windows/win32/seccrypto/example-c-program--verifying-the-signature-of-a-pe-file
bool VerifySignature(const wchar_t* fullPathToFile)
{
    bool valid = true;
    LONG lStatus = {};

    // Initialize the WINTRUST_FILE_INFO structure.
    WINTRUST_FILE_INFO FileData;
    memset(&FileData, 0, sizeof(FileData));
    FileData.cbStruct = sizeof(WINTRUST_FILE_INFO);
    FileData.pcwszFilePath = fullPathToFile;
    FileData.hFile = NULL;
    FileData.pgKnownSubject = NULL;

    if (!pfnWinVerifyTrust)
    {
        // We only support Win10+ so we can search for module in system32 directly
        auto hModWintrust = LoadLibraryExW(L"wintrust.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!hModWintrust || !GetProc(hModWintrust, "WinVerifyTrust", pfnWinVerifyTrust))
        {
            return false;
        }
    }

    /*
    WVTPolicyGUID specifies the policy to apply on the file
    WINTRUST_ACTION_GENERIC_VERIFY_V2 policy checks:

    1) The certificate used to sign the file chains up to a
    * root
    certificate located in the trusted root certificate store. This
    implies that the identity of the publisher has been verified by
    a certification authority.


    * 2) In cases where user interface is displayed (which this example
    does not do), WinVerifyTrust will check for whether the
    end entity certificate is stored in the
    * trusted publisher store,
    implying that the user trusts content from this publisher.

    3) The end entity certificate has sufficient permission to sign
    code, as
    * indicated by the presence of a code signing EKU or no
    EKU.
    */

    GUID WVTPolicyGUID = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    WINTRUST_DATA WinTrustData;

    // Initialize the WinVerifyTrust input data structure.
    // Default all fields to 0.
    memset(&WinTrustData, 0, sizeof(WinTrustData));

    WinTrustData.cbStruct = sizeof(WinTrustData);
    // Use default code signing EKU.
    WinTrustData.pPolicyCallbackData = NULL;
    // No data to pass to SIP.
    WinTrustData.pSIPClientData = NULL;
    // Disable WVT UI.
    WinTrustData.dwUIChoice = WTD_UI_NONE;
    // No revocation checking.
    WinTrustData.fdwRevocationChecks = WTD_REVOKE_NONE;
    // Verify an embedded signature on a file.
    WinTrustData.dwUnionChoice = WTD_CHOICE_FILE;
    // Verify action.
    WinTrustData.dwStateAction = WTD_STATEACTION_VERIFY;
    // Verification sets this value.
    WinTrustData.hWVTStateData = NULL;
    // Not used.
    WinTrustData.pwszURLReference = NULL;
    // This is not applicable if there is no UI because it changes
    // the UI to accommodate running applications instead of
    // installing applications.
    WinTrustData.dwUIContext = 0;
    // Set pFile.
    WinTrustData.pFile = &FileData;

    WINTRUST_SIGNATURE_SETTINGS SignatureSettings = {};
    CERT_STRONG_SIGN_PARA StrongSigPolicy = {};
    SignatureSettings.cbStruct = sizeof(WINTRUST_SIGNATURE_SETTINGS);
    SignatureSettings.dwFlags = WSS_GET_SECONDARY_SIG_COUNT | WSS_VERIFY_SPECIFIC;
    SignatureSettings.dwIndex = 0;
    WinTrustData.pSignatureSettings = &SignatureSettings;

    StrongSigPolicy.cbSize = sizeof(CERT_STRONG_SIGN_PARA);
    StrongSigPolicy.dwInfoChoice = CERT_STRONG_SIGN_OID_INFO_CHOICE;
    StrongSigPolicy.pszOID = (LPSTR)szOID_CERT_STRONG_SIGN_OS_CURRENT;
    WinTrustData.pSignatureSettings->pCryptoPolicy = &StrongSigPolicy;

    // WinVerifyTrust verifies signatures as specified by the GUID  and Wintrust_Data.
    lStatus = pfnWinVerifyTrust(NULL, &WVTPolicyGUID, &WinTrustData);
    valid = lStatus == ERROR_SUCCESS;

    // Any hWVTStateData must be released by a call with close.
    WinTrustData.dwStateAction = WTD_STATEACTION_CLOSE;
    lStatus = pfnWinVerifyTrust(NULL, &WVTPolicyGUID, &WinTrustData);

    return valid;
}

} // namespace security

} // namespace nrc